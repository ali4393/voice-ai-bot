import express from "express";
import path from "path";
import { fileURLToPath } from "url";
import { ChatOpenAI } from "@langchain/openai";
import { StateGraph, START, END, MessagesAnnotation } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { z } from "zod";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(express.static(__dirname));

const baseModel = new ChatOpenAI({
  model: "meta-llama/llama-3.1-8b-instruct",
  apiKey: process.env.OPENROUTER_API_KEY,
  temperature: 0.7,
  maxTokens: 1024,
  configuration: {
    baseURL: "https://openrouter.ai/api/v1",
    defaultHeaders: { "HTTP-Referer": "http://localhost:3000", "X-Title": "Voice AI Chatbot" }
  }
});

const webSearchTool = new DynamicStructuredTool({
  name: "web_search",
  description: "Search the internet for real-time information.",
  schema: z.object({ query: z.string().describe("The search query") }),
  func: async ({ query }) => {
    try {
      const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
      const response = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0" } });
      const html = await response.text();
      const snippets = [];
      const regex = /<a class="result__snippet[^>]*>(.*?)<\/a>/gi;
      let match;
      while ((match = regex.exec(html)) !== null) {
        snippets.push(match[1].replace(/<[^>]+>/g, '').trim());
      }
      return snippets.slice(0, 3).join("\n\n") || "No results found.";
    } catch (error) { return `Search failed: ${error.message}`; }
  }
});

// Define AI Personalities
const personas = {
  default: "You are a friendly, helpful AI assistant. Keep responses short and conversational. Use Markdown for formatting.",
  coder: "You are an expert software engineer. You write clean, efficient code and explain technical concepts clearly. Always use code blocks.",
  teacher: "You are a patient, encouraging tutor. You explain complex topics simply and use examples to help the user learn.",
  professional: "You are a concise, professional business assistant. You get straight to the point, use formal language, and focus on productivity."
};

async function startServer() {
  const allTools = [webSearchTool];
  const modelWithTools = baseModel.bindTools(allTools);
  const toolNode = new ToolNode(allTools);

  let currentLanguage = "English"; 

  async function callModel(state) {
    const systemPrompt = new SystemMessage(
      `You are a helpful AI. You MUST reply strictly in ${currentLanguage}. ` +
      "Keep responses short and conversational. " +
      "For greetings, respond naturally without using tools. " +
      "Only use web_search for news, current events, or specific facts. " +
      "Format your responses beautifully using Markdown."
    );
    const response = await modelWithTools.invoke([systemPrompt, ...state.messages]);
    return { messages: [response] };
  }

  function shouldContinue(state) {
    const lastMessage = state.messages[state.messages.length - 1];
    if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) return "tools";
    return END;
  }

  const workflow = new StateGraph(MessagesAnnotation)
    .addNode("agent", callModel)
    .addNode("tools", toolNode)
    .addEdge(START, "agent")
    .addConditionalEdges("agent", shouldContinue, ["tools", END])
    .addEdge("tools", "agent");

  const agent = workflow.compile();
  let conversationHistory = [new SystemMessage("You are a helpful voice AI assistant.")];

  app.post("/api/chat", async (req, res) => {
    const userMessage = req.body.message;
    const language = req.body.language;
    const persona = req.body.persona || 'default'; // Get persona from frontend
    
    if (language) currentLanguage = language; 

    if (!userMessage) return res.status(400).json({ success: false, error: "Empty message." });

    if (conversationHistory.length > 11) {
      conversationHistory = [conversationHistory[0], ...conversationHistory.slice(-10)];
    }
    conversationHistory.push(new HumanMessage(userMessage));

    try {
      // We temporarily inject the persona into the system prompt for this turn
      const activeSystemPrompt = new SystemMessage(
        `${personas[persona]} You MUST reply strictly in ${currentLanguage}.`
      );
      
      // Create a temporary history with the active persona prompt
      const tempHistory = [activeSystemPrompt, ...conversationHistory.slice(1)];

      const responseState = await agent.invoke({ messages: tempHistory });
      conversationHistory = responseState.messages;
      
      const lastMessage = conversationHistory[conversationHistory.length - 1];
      let reply = "";
      
      if (typeof lastMessage.content === "string") reply = lastMessage.content;
      else if (Array.isArray(lastMessage.content)) reply = lastMessage.content.map(c => c.text || "").join("");
      
      res.json({ success: true, reply: reply });
    } catch (err) {
      console.error("❌ [ERROR]:", err.message);
      res.status(500).json({ success: false, error: err.message });
    }
  });

  app.post("/api/reset", (req, res) => {
    conversationHistory = [new SystemMessage("You are a helpful voice AI assistant.")];
    res.json({ success: true });
  });

  app.listen(PORT, () => {
    console.log(`🚀 Server running on port ${PORT}`);
  });
}

startServer().catch(err => { console.error("Failed to start:", err); process.exit(1); });