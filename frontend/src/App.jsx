import React, { useState } from 'react'
import UploadDropzone from './components/UploadDropzone.jsx'
import ResultViewer from './components/ResultViewer.jsx'
import { postProcess } from './api.js'

export default function App() {
  const [files, setFiles] = useState([])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleRun() {
    setLoading(true); setError(''); setResult(null)
    try {
      const res = await postProcess(files)
      setResult(res)
    } catch (e) {
      setError(e.message || '처리 실패')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>반사광 제거형 OCR 스캐너</h1>
      <p style={{opacity:.8, marginBottom:16}}>
        사진 속 반사광을 제거하고 문서를 보정한 뒤, 한/영 OCR로 텍스트를 추출합니다.
        여러 장을 올리면 글레어 없는 프레임을 병합합니다.
      </p>
      <div className="card" style={{marginBottom:16}}>
        <UploadDropzone onFiles={setFiles}/>
        {files?.length > 0 && (
          <p style={{marginTop:8}} className="badge">{files.length}개 파일 선택됨</p>
        )}
        <button className="btn" style={{marginTop:14}} onClick={handleRun} disabled={!files.length || loading}>
          {loading ? '처리 중…' : '처리 시작'}
        </button>
        {error && <p style={{color:'#ff6b6b', marginTop:10}}>{error}</p>}
      </div>
      <ResultViewer result={result}/>
      <footer style={{opacity:.5, marginTop:24}}>Made with React + FastAPI + OpenCV + PaddleOCR</footer>
    </div>
  )
}
