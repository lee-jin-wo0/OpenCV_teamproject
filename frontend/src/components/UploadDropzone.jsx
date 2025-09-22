import React, { useState } from 'react'

export default function UploadDropzone({ onFiles }) {
  const [drag, setDrag] = useState(false)
  return (
    <div className="dropzone"
      onDragOver={e => {e.preventDefault(); setDrag(true)}}
      onDragLeave={() => setDrag(false)}
      onDrop={e => {
        e.preventDefault(); setDrag(false)
        const files = Array.from(e.dataTransfer.files)
        onFiles(files)
      }}>
      <p style={{opacity:.8}}>이미지 파일을 드래그&드롭하거나</p>
      <input id="file" type="file" accept="image/*" multiple
             style={{display:'none'}}
             onChange={e => onFiles(Array.from(e.target.files))}/>
      <label htmlFor="file" className="btn" style={{display:'inline-block', marginTop:12}}>
        파일 선택
      </label>
      {drag && <p style={{marginTop:8}} className="badge">놓으면 업로드됩니다</p>}
    </div>
  )
}
