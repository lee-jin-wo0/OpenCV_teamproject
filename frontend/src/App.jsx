import React, { useState, useEffect } from 'react'
import UploadDropzone from './components/UploadDropzone.jsx'
import ResultViewer from './components/ResultViewer.jsx'
import Modal from './components/Modal.jsx'
import { postProcess } from './api.js'
import './styles.css'

export default function App() {
  const [files, setFiles] = useState([])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [open, setOpen] = useState(false)

  async function handleRun() {
    setLoading(true); setError(''); setResult(null)
    try {
      const res = await postProcess(files)
      setResult(res)
      setOpen(true)             // ✅ 처리 완료 시 모달 열기
    } catch (e) {
      setError(e.message || '처리 실패')
    } finally {
      setLoading(false)
    }
  }

  // ESC로 닫기
  useEffect(() => {
    const onKey = (e)=>{ if(e.key==='Escape') setOpen(false) }
    window.addEventListener('keydown', onKey); return ()=>window.removeEventListener('keydown', onKey)
  },[])

  return (
    <div className="container blobs">
      {/* Hero */}
      <section className="hero">
        <div className="kicker">GLARE-FREE DOC SCANNER</div>
        <h1 className="headline">
          Make your<br/> <span className="accent">scanning<br/></span> more convenient.
        </h1>
        <p className="sub">
          사진 속 반사광을 제거하고 문서를 보정한 뒤, 한/영 OCR로 텍스트를 추출합니다.<br/>
        </p>
      </section>

      {/* Upload Card */}
      <section className="card" style={{ margin: '22px auto 26px', maxWidth: 820 }}>
        <div className="stack" style={{ alignItems: 'center' }}>
          <UploadDropzone onFiles={setFiles} />
          {files?.length > 0 && <p className="badge">{files.length}개 파일 선택됨</p>}
          <button className="btn" onClick={handleRun} disabled={!files.length || loading}>
            {loading ? '처리 중…' : '처리 시작'}
          </button>
          {error && <p style={{ color:'#ff6b6b', marginTop:6 }}>{error}</p>}
        </div>
      </section>

      {/* Result Modal */}
      <Modal open={open} onClose={()=>setOpen(false)} title="처리 결과">
        <ResultViewer result={result} />
      </Modal>

      <footer>Made with React · FastAPI · OpenCV · PaddleOCR</footer>
    </div>
  )
}
