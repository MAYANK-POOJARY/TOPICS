import "dotenv/config"
import fs from 'fs';
import { PDFParse } from 'pdf-parse';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { MistralAIEmbeddings } from "@langchain/mistralai";
import { Pinecone } from '@pinecone-database/pinecone';

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY
});

const index = pc.index({ host: process.env.PINECONE_HOST, namespace: '__default__' });

// const info = await pc.describeIndex("rag");
// console.log(JSON.stringify(info, null, 2));

// model for embedding
const embeddings = new MistralAIEmbeddings({
    apiKey:process.env.MISTRAL_API_KEY,
    model: "mistral-embed"
});


// reads the file and returns the pdf data in the form of buffer as the data in pdf is not present in plain text but in binary format 
// const dataBuffer = fs.readFileSync('./story.pdf');

// // pass the buffer data to the pdfparser 
// const parser = new PDFParse({
//     data: dataBuffer
// })

// get the text data from the pdf based on pages
// const result = await parser.getText()

// textsplitter is used to split the text and the chunksize is determined 
// const splitter = new RecursiveCharacterTextSplitter({
//     chunkSize:500,
//     chunkOverlap:0
// })

// // the data is splitted in chunks 
// const chunks = await splitter.splitText(result.text);


// embedding is performed 
// const docs = await Promise.all(chunks.map(async (chunk) =>{
//     const embeddedData = await embeddings.embedQuery(chunk);
//     return{
//         text: chunk,
//         embeddedData
//     }
// }))

//old
// const resultData = await index.upsert({
//     records: docs.map((doc, i) =>({
//         id: `doc-${i}`,
//         values: doc.embeddedData,
//         metadata:{
//             text: doc.text
//         }
//     }))
// })

// const resultData = await index.documents.upsert({
//      documents: docs.map((doc, i) => ({
//        _id: `doc-${i}`,
//        embedding: doc.embeddedData,   // your dense_vector field name
//        text: doc.text
//      }))
//    });

// console.log(resultData)

// Pinecone allows at most 96 documents per upsert with integrated embedding
// for (let i = 0; i < chunks.length; i += 96) {
//   const batch = chunks.slice(i, i + 96);
//   await index.documents.upsert({
//     documents: batch.map((text, j) => ({
//       _id: `doc-${i + j}`,
//       title: 'story.pdf',
//       chunk_text: text
//     }))
//   });
// }

// console.log(`Upserted ${chunks.length} chunks`);


// question asked for getting the answer 
// const queryEmbedding = await embeddings.embedQuery("How was the internship experience ?");

// const result = await index.query({
//     vector: queryEmbedding,
//     topK: 2,
//     includeMetadata: true
// });

// console.log(JSON.stringify(result))

const response = await index.documents.search({
  topK: 2,
  scoreBy: [
    { type: 'embed', fields: ['chunk_text'], query: 'How was the internship experience?' }
  ],
  includeFields: ['chunk_text']
});

console.log(JSON.stringify(response, null, 2));