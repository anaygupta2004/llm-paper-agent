const API_BASE_URL = import.meta.env.PROD
  ? 'https://arxiv-agent.vercel.app/api'
  : '/api';

async function handleResponse(response: Response) {
  if (!response.ok) {
    if (response.status >= 500) {
      throw new Error(`${response.status}: ${response.statusText}`);
    }

    const errorText = await response.text();
    throw new Error(`${response.status}: ${errorText}`);
  }

  return response.json();
}

export async function fetchWithAuth(
  endpoint: string,
  options: RequestInit = {}
) {
  const url = `${API_BASE_URL}${endpoint}`;
  console.debug('[API] Request:', { url, method: options.method || 'GET' });

  // Get the token from Firebase Auth if available
  const token = await getAuthToken();

  const headers: Record<string, string> = {
    ...options.headers as Record<string, string>,
    'Content-Type': 'application/json',
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  try {
    const response = await fetch(url, {
      ...options,
      credentials: 'include',
      headers,
    });

    const result = await handleResponse(response);
    console.debug('[API] Response:', { url, status: response.status, ok: response.ok });
    return result;
  } catch (error) {
    console.error('[API] Error:', { url, error });
    throw error;
  }
}

async function getAuthToken(): Promise<string | null> {
  try {
    const { auth } = await import('@/lib/firebase');
    const user = auth.currentUser;
    if (!user) return null;
    return user.getIdToken();
  } catch (error) {
    console.error('[Auth] Error getting token:', error);
    return null;
  }
}

export const apiClient = {
  get: (endpoint: string) => fetchWithAuth(endpoint),
  post: (endpoint: string, data: any) => fetchWithAuth(endpoint, {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  put: (endpoint: string, data: any) => fetchWithAuth(endpoint, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),
  delete: (endpoint: string) => fetchWithAuth(endpoint, {
    method: 'DELETE',
  }),
};