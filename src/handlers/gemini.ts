import {
  EnhancedGenerateContentResponse,
  GoogleGenerativeAI,
} from '@google/generative-ai';
import {
  HandlerParams,
  HandlerParamsNotStreaming,
  HandlerParamsStreaming,
  ResultNotStreaming,
  ResultStreaming,
  FinishReason,
} from '../types';

function mapFinishReason(geminiReason: string | undefined): FinishReason {
  switch (geminiReason) {
    case 'STOP':
      return 'stop';
    case 'MAX_TOKENS':
      return 'length';
    case 'SAFETY':
    case 'RECITATION':
      return 'content_filter';
    default:
      return 'stop';
  }
}

async function* toStreamingResponse(
  response: AsyncIterable<EnhancedGenerateContentResponse>,
): ResultStreaming {
  for await (const chunk of response) {
    yield {
      model: chunk.candidates?.[0]?.content?.role ?? 'model',
      created: Date.now(),
      choices: [
        {
          delta: {
            content: chunk.candidates?.[0]?.content?.parts?.[0]?.text ?? '',
            role: chunk.candidates?.[0]?.content?.role ?? 'model',
          },
          index: 0,
          finish_reason: mapFinishReason(chunk.candidates?.[0]?.finishReason),
        },
      ],
    };
  }
}

export async function GeminiHandler(
  params: HandlerParamsNotStreaming,
): Promise<ResultNotStreaming>;

export async function GeminiHandler(
  params: HandlerParamsStreaming,
): Promise<ResultStreaming>;

export async function GeminiHandler(
  params: HandlerParams,
): Promise<ResultNotStreaming | ResultStreaming>;

export async function GeminiHandler(
  params: HandlerParams,
): Promise<ResultNotStreaming | ResultStreaming> {
  const { apiKey: providedApiKey, ...completionsParams } = params;
  const apiKey = providedApiKey ?? process.env.GEMINI_API_KEY;

  if (!apiKey) {
    throw new Error('No API key provided');
  }

  const genAI = new GoogleGenerativeAI(apiKey);
  const model = genAI.getGenerativeModel({ model: params.model });

  // if there's a "system" prompt in the messages, use it as the system instruction
  const systemInstruction = completionsParams.messages.find(
    (msg) => msg.role === 'system',
  )?.content;

  // remove the "system" prompt from the messages
  const messages = completionsParams.messages.filter(
    (msg) => msg.role !== 'system',
  );

  const chat = model.startChat({
    systemInstruction: systemInstruction ?? undefined,
    history: messages.slice(0, -1).map((msg) => ({
      role: msg.role,
      parts: [{ text: msg.content ?? '' }],
    })),
    generationConfig: {
      temperature: completionsParams.temperature ?? undefined,
      topP: completionsParams.top_p ?? undefined,
      maxOutputTokens: completionsParams.max_tokens ?? undefined,
    },
  });

  const lastMessage =
    completionsParams.messages[completionsParams.messages.length - 1];

  if (!lastMessage.content) {
    throw new Error('No content provided');
  }

  if (params.stream) {
    const response = await chat.sendMessageStream(lastMessage.content);
    return toStreamingResponse(response.stream);
  }

  const response = await chat.sendMessage(lastMessage.content);

  return {
    model: params.model,
    created: Date.now(),
    choices: [
      {
        index: 0,
        message: {
          role: 'model',
          content: response.response.text(),
        },
        finish_reason:
          mapFinishReason(response.response.candidates?.[0]?.finishReason) ??
          'stop',
      },
    ],
  };
}
