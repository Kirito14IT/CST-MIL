export async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...options,
      headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
    });
  } catch {
    throw new Error('Cannot connect to the local service. Check that the launcher is running, then retry.');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error([body?.error || `Request failed (${response.status})`, ...(body?.details || [])].join('；'));
  }
  return response.json() as Promise<T>;
}

export const post = <T,>(path: string, data: unknown) => request<T>(path, { method: 'POST', body: JSON.stringify(data) });
export const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'The operation did not finish. Please retry.';
