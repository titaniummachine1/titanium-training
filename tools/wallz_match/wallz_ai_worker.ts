/// <reference lib="webworker" />
import { findBestMove } from '@wallz/ai';
import type { GameState, Move } from '@wallz/game';

export type AiRequest = {
  id: number;
  state: GameState;
  maxDepth?: number;
  timeMs?: number;
  wallBias?: number;
  blunderRate?: number;
  blunderMargin?: number;
};
export type AiResponse =
  | { id: number; move: Move }
  | { id: number; error: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.addEventListener('message', (e: MessageEvent<AiRequest>) => {
  const { id, state, maxDepth = 2, timeMs = 350, wallBias = 1, blunderRate = 0, blunderMargin = 0 } = e.data;
  try {
    const move = findBestMove(state, { maxDepth, timeMs, wallBias, blunderRate, blunderMargin });
    ctx.postMessage({ id, move } satisfies AiResponse);
  } catch (err) {
    // Never swallow a throw silently, that leaves the requestMove promise
    // unresolved and the vs-AI loop stuck "thinking" forever. Report it back so
    // the caller can reject and recover.
    const message = err instanceof Error ? err.message : String(err);
    ctx.postMessage({ id, error: message } satisfies AiResponse);
  }
});
