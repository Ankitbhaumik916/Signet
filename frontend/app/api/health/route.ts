import { NextResponse } from 'next/server'

/**
 * GET /api/health
 * Proxies health request to Python FastAPI backend
 */
export async function GET() {
  try {
    const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000'
    const backendResponse = await fetch(`${backendUrl}/health`, {
      method: 'GET',
      cache: 'no-store',
    })

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({ detail: 'Backend error occurred' }))
      return NextResponse.json(
        { error: errorData.detail || 'Failed to load health status' },
        { status: backendResponse.status }
      )
    }

    const payload = await backendResponse.json()
    return NextResponse.json(payload, { status: 200 })
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
    return NextResponse.json(
      {
        error: 'Failed to connect to health service',
        details: errorMessage,
      },
      { status: 500 }
    )
  }
}
