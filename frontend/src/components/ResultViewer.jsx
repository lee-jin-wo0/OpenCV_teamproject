import React from 'react'

export default function ResultViewer({ result }) {
  if (!result) return null
  const { merged_used, text, processed_image_b64, boxes } = result
  return (
    <div className="grid">
      <div className="card">
        <h3>보정 결과</h3>
        <img src={processed_image_b64} alt="processed" style={{width:'100%', borderRadius:12}}/>
        <p style={{marginTop:8}}>
          병합 사용: <span className="badge">{merged_used ? '여러 프레임 병합' : '단일 프레임'}</span>
        </p>
      </div>
      <div className="card">
        <h3>추출 텍스트</h3>
        <pre>{text || '(인식된 텍스트 없음)'}</pre>
        <details style={{marginTop:12}}>
          <summary>디텍션 박스({boxes?.length ?? 0})</summary>
          <ul>
            {boxes?.map((b,i)=>(
              <li key={i} style={{marginBottom:6}}>
                #{i+1} [{b.conf.toFixed(2)}] {b.text}
              </li>
            ))}
          </ul>
        </details>
        {text && (
          <button className="btn" style={{marginTop:12}} onClick={()=>{
            const blob = new Blob([text], {type:'text/plain;charset=utf-8'})
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url; a.download = 'extracted_text.txt'; a.click()
            URL.revokeObjectURL(url)
          }}>텍스트 다운로드</button>
        )}
      </div>
    </div>
  )
}
