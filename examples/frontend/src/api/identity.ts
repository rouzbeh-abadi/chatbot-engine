/**
 * Who this browser is, for the purpose of long-term memory.
 *
 * Memory follows a person rather than a conversation, so it needs an id that
 * outlives a chat. This one is generated once and kept in `localStorage`, so it
 * survives New chat, a reload, and a restart, and two browsers stay separate.
 *
 * It is a partition, not a login. Anyone can set this header to anyone's id, so
 * it separates people by cooperation rather than by enforcement. That is only
 * acceptable because the example backend authenticates nobody; with
 * BACKEND_TRUST_USER_HEADER on, the backend ignores this and uses the
 * authenticated user instead.
 */
const KEY = "chatbot-engine.user-id";

let cached: string | null = null;

export function userId(): string {
  if (cached) return cached;

  try {
    const stored = localStorage.getItem(KEY);
    if (stored) {
      cached = stored;
      return stored;
    }
    const fresh = crypto.randomUUID();
    localStorage.setItem(KEY, fresh);
    cached = fresh;
    return fresh;
  } catch {
    // Private browsing, or storage disabled. Memory then lasts as long as the
    // tab, which is worse than intended but better than failing to chat.
    cached ??= crypto.randomUUID();
    return cached;
  }
}

/** Start again as someone new, forgetting everything stored about this browser. */
export function resetUserId(): string {
  cached = null;
  try {
    localStorage.removeItem(KEY);
  } catch {
    // Nothing to clear; `cached` was the only copy.
  }
  return userId();
}

/** The identity header every request that touches memory must carry. */
export function identityHeaders(): Record<string, string> {
  return { "X-User-Id": userId() };
}
