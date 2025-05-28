import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'

function App() {

    const [grade, setGrade] = useState(1);
    const [angle, setAngle] = useState(0);

    const [img, setImg] = useState(null);

    const gradeOptions = [
      { label: "V1", value: 1 },
      { label: "V2", value: 2 },
      { label: "V3", value: 3 },
      { label: "V4", value: 4 },
      { label: "V5", value: 5 },
      { label: "V6", value: 6 },
      { label: "V7", value: 7 },
      { label: "V8", value: 8 },
      { label: "V9", value: 9 },
      { label: "V10", value: 10 },
      { label: "V11", value: 11 },
      { label: "V12", value: 12 },
      { label: "V13", value: 13 },
      { label: "V14", value: 14 },
      { label: "V15", value: 15 },
      { label: "V16", value: 16 },
      { label: "V17", value: 17 }
    ];
    const angleOptions = [
      { label: "0", value: 0 },
      { label: "5", value: 5 },
      { label: "10", value: 10 },
      { label: "15", value: 15 },
      { label: "20", value: 20 },
      { label: "25", value: 25 },
      { label: "30", value: 30 },
      { label: "35", value: 35 },
      { label: "40", value: 40 },
      { label: "45", value: 45 },
      { label: "50", value: 50 },
      { label: "55", value: 55 },
      { label: "60", value: 60 },
      { label: "65", value: 65 },
      { label: "70", value: 70 }
    ];

    const handleChangeGrade = (event) => {
        setGrade(event.target.value);
    }
    const handleChangeAngle = (event) => {
        setAngle(event.target.value);
    }

    const handleGenerate = async () => {
        // setLoading(true);
        console.log('grade', grade, 'angle', angle)
        try {
            console.log("Generating...");

            const resp = await fetch('/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    grade: grade,
                    angle: angle
                })
            });

            if (!resp.ok) {
                throw new Error(`HTTP error! Status: ${resp.status}`);
            }

            console.log("response received");
            const imageBlob = await resp.blob();
            const url = URL.createObjectURL(imageBlob);
            setImg(url);
        } catch (error) {
            console.error("Generation failed:", error);
        } finally {
          // setLoading(false);
        }
    }

    return (
        <>
            <div>
                <a href="https://vite.dev" target="_blank">
                    <img src={viteLogo} className="logo" alt="Vite logo" />
                </a>
                <a href="https://react.dev" target="_blank">
                    <img src={reactLogo} className="logo react" alt="React logo" />
                </a>
            </div>
            <h1 className="header">setter</h1>
            <div className="card">
                <label>
                    Grade:
                    <select value={grade} onChange={handleChangeGrade}>
                        {gradeOptions.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                    </select>
                </label>
                <label>
                    Angle:
                    <select value={angle} onChange={handleChangeAngle}>
                        {angleOptions.map((option) => (
                            <option key={option.value} value={option.value}>{option.label}</option>
                        ))}
                    </select>
                </label>
            </div>
            <button onClick={handleGenerate}>
                Generate
            </button>
            <img src={img}/>
        </>
    )
}

export default App
