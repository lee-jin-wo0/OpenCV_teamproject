import React, { useState } from 'react'

export default function ResultViewer({ result }) {
  const [active, setActive] = useState('image') // 'image' | 'text'
  if (!result) return <div className="panel panel--image">결과를 준비하는 중…</div>

  const { merged_used, text, processed_image_b64, boxes } = result

  return (
    <>
      {/* 모바일 탭 */}
      <div className="tabs">
        <button className="tab" aria-selected={active==='image'} onClick={()=>setActive('image')}>보정 결과</button>
        <button className="tab" aria-selected={active==='text'} onClick={()=>setActive('text')}>추출 텍스트</button>
      </div>

      <div className="modal-grid">
        <span className="v-sep" aria-hidden="true"></span>

        {/* Left: Image Panel */}
        <section
          className="panel panel--image"
          style={{ display: (active==='image'|| window.innerWidth>930) ? 'block':'none' }}
        >
          <header className="panel__head">
            <h3 className="panel__title">보정 결과</h3>
            <span className="badge panel__sub">
              {merged_used ? '여러 프레임 병합' : '단일 프레임'}
            </span>
          </header>

          <img src={processed_image_b64} alt="processed" />
          <div className="panel__toolbar">
            {/* ✅ 통일: 텍스트 다운로드와 같은 Primary 버튼 */}
            <a
              className="btn"
              href={processed_image_b64}
              download="processed_image.png"
            >
              이미지 다운로드
            </a>
            <button className="btn btn-ghost" onClick={()=>window.open(processed_image_b64, '_blank')}>
              새 창으로 보기
            </button>
          </div>
        </section>

        {/* Right: Text Panel */}
        <section
          className="panel panel--text"
          style={{ display: (active==='text'|| window.innerWidth>930) ? 'block':'none' }}
        >
          <header className="panel__head">
            <h3 className="panel__title">추출 텍스트</h3>
            <span className="badge panel__sub">박스 {boxes?.length ?? 0}개</span>
          </header>

          <div className="panel__box">
            <pre style={{ margin:0 }}>{text || '(인식된 텍스트 없음)'}</pre>
          </div>

          <details style={{ marginTop:12 }}>
            <summary>디텍션 박스 목록</summary>
            <ul style={{ marginTop:8 }}>
              {boxes?.map((b,i)=>(
                <li key={i} style={{ marginBottom:6 }}>
                  #{i+1} [{b.conf.toFixed(2)}] {b.text}
                </li>
              ))}
            </ul>
          </details>

          <div className="panel__toolbar">
            <button
              className="btn"
              onClick={() => {
                const blob = new Blob([text || ''], { type:'text/plain;charset=utf-8' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url; a.download = 'extracted_text.txt'; a.click()
                URL.revokeObjectURL(url)
              }}
              disabled={!text}
            >
              텍스트 다운로드
            </button>
            <button
              className="btn btn-ghost"
              onClick={async ()=>{
                if (!text) return
                try{
                  await navigator.clipboard.writeText(text)
                  alert('텍스트를 클립보드에 복사했습니다.')
                }catch{
                  const ta = document.createElement('textarea')
                  ta.value = text; document.body.appendChild(ta)
                  ta.select(); document.execCommand('copy'); document.body.removeChild(ta)
                  alert('텍스트를 클립보드에 복사했습니다.')
                }
              }}
              disabled={!text}
            >
              복사하기
            </button>
          </div>
        </section>
      </div>
    </>
  )
}
