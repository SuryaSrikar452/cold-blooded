/**
 * Gemini Service - Handles low-level communication with the Gemini API.
 * Trust Boundary: Reads GEMINI_API_KEY strictly from process.env.
 * Does not contain domain-specific cohort logic.
 */

require('dotenv').config();

const CANDIDATE_MODELS = [
  'gemini-3.5-flash',
  'gemini-3.6-flash',
  'gemini-3.7-flash',
  'gemini-flash-latest'
];

/**
 * Generate structured content using Gemini.
 * @param {string} systemInstruction System prompt defining behavior/schema
 * @param {string} userPrompt The user prompt or contextual request
 * @returns {Promise<any>} Parsed JSON response from Gemini
 */
async function generateStructuredContent(systemInstruction, userPrompt) {
  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey || apiKey === 'YOUR_REAL_KEY_HERE' || apiKey === 'your_gemini_api_key_here') {
    throw new Error('GEMINI_API_KEY is not configured in backend/.env. Please provide a valid Gemini API key.');
  }

  let lastError = null;

  // Try candidate models in order of reliability
  for (const modelName of CANDIDATE_MODELS) {
    try {
      // 1. Attempt using official Google GenAI SDK
      try {
        const { GoogleGenAI } = require('@google/genai');
        const ai = new GoogleGenAI({ apiKey });
        const response = await ai.models.generateContent({
          model: modelName,
          contents: userPrompt,
          config: {
            systemInstruction,
            responseMimeType: 'application/json',
            temperature: 0.1,
          },
        });

        const text = response.text || (response.candidates?.[0]?.content?.parts?.[0]?.text);
        if (text) {
          return JSON.parse(text);
        }
      } catch (sdkErr) {
        // Continue to native REST API for this model
      }

      // 2. Native HTTPS REST call fallback
      const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey}`;
      const requestBody = {
        contents: [
          {
            role: 'user',
            parts: [{ text: userPrompt }]
          }
        ],
        systemInstruction: {
          parts: [{ text: systemInstruction }]
        },
        generationConfig: {
          responseMimeType: 'application/json',
          temperature: 0.1
        }
      };

      const res = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        const msg = errData?.error?.message || `HTTP ${res.status}`;
        throw new Error(msg);
      }

      const data = await res.json();
      const candidateText = data?.candidates?.[0]?.content?.parts?.[0]?.text;
      if (candidateText) {
        return JSON.parse(candidateText);
      }
    } catch (err) {
      lastError = err;
      // If overloaded/not found, try next candidate model
      continue;
    }
  }

  throw new Error(`Gemini API error: ${lastError ? lastError.message : 'No candidate model responded'}`);
}

module.exports = {
  generateStructuredContent,
};
