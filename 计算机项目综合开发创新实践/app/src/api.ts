export async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...options,
      headers: { ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
    });
  } catch {
    throw new Error('暂时无法连接本机服务。请确认启动窗口仍在运行，然后重试。');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error([body?.error || `请求未完成（${response.status}）`, ...(body?.details || [])].join('；'));
  }
  return response.json() as Promise<T>;
}

export const post = <T,>(path: string, data: unknown) => request<T>(path, { method: 'POST', body: JSON.stringify(data) });
export const errorMessage = (error: unknown) => error instanceof Error ? error.message : '操作未完成，请重试。';
