import { useState } from 'react';
import { parseCSV } from '../utils/fileParser';

interface UploadSectionProps {
  onAnalyze: (data: any[]) => void;
}

export default function UploadSection({ onAnalyze }: UploadSectionProps) {
  const [dragActive, setDragActive] = useState(false);

  const handleDrop = async (e) => {
    e.preventDefault();
    setDragActive(false);
    
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'text/csv') {
      const data = await parseCSV(file);
      onAnalyze(data);
    }
  };

  const handleFileInput = async (e) => {
    const file = e.target.files[0];
    if (file) {
      const data = await parseCSV(file);
      onAnalyze(data);
    }
  };

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4">Upload Your Transactions</h2>
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center ${
          dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
        }`}
        onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
      >
        <input
          type="file"
          accept=".csv"
          onChange={handleFileInput}
          className="hidden"
          id="file-upload"
        />
        <label
          htmlFor="file-upload"
          className="cursor-pointer text-blue-600 hover:text-blue-800"
        >
          Click to upload
        </label>
        <p className="mt-2 text-gray-600">or drag and drop your CSV file here</p>
      </div>
    </div>
  );
}
