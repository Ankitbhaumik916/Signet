import { NextRequest, NextResponse } from 'next/server'

/**
 * POST /api/verify
 * Proxies signature verification request to Python FastAPI backend
 */
export async function POST(request: NextRequest) {
  try {
    const backendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000'
    
    const formData = await request.formData()
    
    // Validate that both images are provided
    const genuineFile = formData.get('genuine')
    const testFile = formData.get('test')
    
    if (!genuineFile || !testFile) {
      return NextResponse.json(
        { error: 'Both genuine and test signature images are required' },
        { status: 400 }
      )
    }
    
    // Create new FormData for backend request
    const backendFormData = new FormData()
    backendFormData.append('genuine', genuineFile)
    backendFormData.append('test', testFile)
    
    // Make request to Python backend
    const backendResponse = await fetch(`${backendUrl}/verify`, {
      method: 'POST',
      body: backendFormData,
      // Don't set Content-Type header - browser will set it with boundary
    })
    
    // Handle backend errors
    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => ({
        detail: 'Backend error occurred'
      }))
      
      return NextResponse.json(
        { error: errorData.detail || 'Verification failed' },
        { status: backendResponse.status }
      )
    }
    
    // Return backend response
    const resultData = await backendResponse.json()
    
    return NextResponse.json(resultData, { status: 200 })
  } catch (error) {
    console.error('API route error:', error)
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
    
    return NextResponse.json(
      {
        error: 'Failed to connect to verification service',
        details: errorMessage
      },
      { status: 500 }
    )
  }
}

/**
 * OPTIONS /api/verify
 * Handle CORS preflight requests
 */
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  })
}
