import "dotenv/config"
import fs from 'fs';
import { PDFParse } from 'pdf-parse';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Pinecone } from '@pinecone-database/pinecone';
import { ChatMistralAI } from "@langchain/mistralai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import path from "path";
import crypto from 'crypto';

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


function generateDocumentId(filePath) {

    const fileBuffer = fs.readFileSync(filePath);

    const documentId = crypto
        .createHash('sha256')
        .update(fileBuffer)
        .digest('hex');

    return documentId;
}


async function ingest(filePath) {

    const fileName = path.basename(filePath);

    const dataBuffer = fs.readFileSync(filePath);

    // Generate ID from the actual PDF contents
    const documentId = generateDocumentId(filePath);

    const parser = new PDFParse({
        data: dataBuffer
    });

    try {

        const result = await parser.getText();

        const pages = result.pages.map((page, index) => ({
            pageNumber: index + 1,
            text: page.text
        }));

        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 500,
            chunkOverlap: 200
        });

        const allChunks = [];

        for (const page of pages) {

            const chunks = await splitter.splitText(page.text);

            chunks.forEach((chunk, index) => {

                if (!chunk.trim()) return;

                allChunks.push({
                    documentId,
                    title: fileName,
                    pageNumber: page.pageNumber,
                    chunkNumber: index + 1,
                    chunkText: chunk
                });

            });
        }


        for (let i = 0; i < allChunks.length; i += 96) {

            const batch = allChunks.slice(i, i + 96);

            await index.documents.upsert({
                documents: batch.map(chunk => ({
                    _id: `${documentId}-p${chunk.pageNumber}-c${chunk.chunkNumber}`,
                    title: chunk.title,
                    document_id: chunk.documentId,
                    chunk_text: chunk.chunkText,
                    page_number: chunk.pageNumber,
                    chunk_number: chunk.chunkNumber
                }))
            });

        }

        console.log(`${fileName}: Upserted ${allChunks.length} chunks`);

    } catch(error){
        console.log("ERROR -->", error);
    }finally {
        await parser.destroy();
    }
}


// input pdf files
await ingest('./aarav_life_1_sisters_wedding.pdf');
await ingest('./aarav_life_3_money_and_friends.pdf');

    const userQuestion = 'Which all documents do you have related to aarav?'

    const response = await index.documents.search({
    topK: 2,
    scoreBy: [
        { type: 'embed', fields: ['chunk_text'], query: userQuestion }
    ],
    includeFields: [
            'chunk_text',
            'title',
            'document_id',
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