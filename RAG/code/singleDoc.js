import "dotenv/config"
import fs from 'fs';
import { PDFParse } from 'pdf-parse';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Pinecone } from '@pinecone-database/pinecone';
import { ChatMistralAI } from "@langchain/mistralai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

// configuration
const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY
});

const index = pc.index({ 
    host: process.env.PINECONE_HOST, 
    namespace: '__default__' 
});

export const mistralModel = new ChatMistralAI({
    model:"ministral-8b-latest",
    apiKey: process.env.MISTRAL_API_KEY
})



// reads the file and returns the pdf data in the form of buffer as the data in pdf is not present in plain text but in binary format 
const dataBuffer = fs.readFileSync('./story.pdf');

// pass the buffer data to the pdfparser 
const parser = new PDFParse({
    data: dataBuffer
})

// get the text data from the pdf based on pages
const result = await parser.getText();

const pages = result.pages.map((page, index) => ({
    pageNumber: index + 1,
    text: page.text
}));

// textsplitter is used to split the text and the chunksize is determined 
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 200
});

// converting the data on each page into chunks of given size
const allChunks = [];

for (const page of pages) {

    const chunks = await splitter.splitText(page.text);

    chunks.forEach((chunk, index) => {
        allChunks.push({
            pageNumber: page.pageNumber,
            chunkNumber: index + 1,
            chunkText: chunk
        });
    });
}

console.log(allChunks);


// Pinecone allows at most 96 documents per upsert with integrated embedding
for (let i = 0; i < allChunks.length; i += 96) {
  const batch = allChunks.slice(i, i + 96);
  await index.documents.upsert({
    documents: batch.map((chunk, j) => ({
        _id: `doc-${i + j}`,
        title: 'story.pdf',
        chunk_text: chunk.chunkText,
        page_number: chunk.pageNumber,
        chunk_number: chunk.chunkNumber
    }))
  });
}

console.log(`Upserted ${allChunks.length} chunks`);

await parser.destroy();

const userQuestion = 'How was the internship experience?'

const response = await index.documents.search({
  topK: 2,
  scoreBy: [
    { type: 'embed', fields: ['chunk_text'], query: userQuestion }
  ],
  includeFields: [
        'chunk_text',
        'page_number',
        'chunk_number'
    ]
});



const finalAnswer = await mistralModel.invoke([
    new SystemMessage(`
        You are an AI assistant responsible for generating
        clear and meaningful answers from retrieved information.

        Your task is to:
        1. Understand the provided text.
        2. Combine the relevant information from both responses.
        3. Convert it into grammatically correct, natural sentences.
        4. Generate the final answer in paragraph format.
        5. Do not add information that is not present in the context.
        6. Do not invent or assume anything.

        Return only the final answer.
    `),
    new HumanMessage(`
        The Question asked by the user : 
        ${userQuestion}

        Retrieved Context:
        ${JSON.stringify(response.matches)}
    `)
])

console.log(finalAnswer.text);
