import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [inputValue, setInputValue] = useState('');
  const [decimalInput, setDecimalInput] = useState('');
  const [response, setResponse] = useState('');
  const [displayData, setDisplayData] = useState(null);
  const [residualPlot, setResidualPlot] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [question, setQuestion] = useState('');
  const [responseCode, setResponseCode] = useState('');
  const [PlotUrl, setPlotUrl] = useState('');
  const [loading, setLoading] = useState(false);

  const handleQuestionChange = (event) => {
    setQuestion(event.target.value);
  };

  const handleGenerateCode = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:5001/generate_code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error('Failed to generate code');
      }

      const data = await response.json();
      setResponseCode(data.code);

      // Send the generated code to the backend for execution
      const execResponse = await fetch('http://localhost:5001/execute_code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ code: data.code }),
      });

      if (!execResponse.ok) {
        throw new Error('Failed to execute code');
      }

      const execData = await execResponse.json();
      setPlotUrl(execData.imageUrl);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  }

  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleDecimalInputChange = (event) => {
    setDecimalInput(event.target.value);
  };

  const handleSubmit = async () => {
    if (!inputValue.trim()) return;

    try {
      const response = await fetch('http://localhost:5001/question', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: inputValue })
      });

      if (!response.ok) {
        throw new Error(`An error has occurred: ${response.status}`);
      }

      const result = await response.json();
      setResponse(result.response);
      setChatHistory(prevHistory => [...prevHistory, { question: inputValue, answer: result.response }]);
      setInputValue('');
    } catch (error) {
      console.error('Error:', error);
      setResponse('Failed to send data to Flask');
    }
  };

  const handleDecimalSubmit = async () => {
    const decimals = decimalInput.split(' ').map(Number);
    if (decimals.length !== 3 || decimals.some(num => isNaN(num) || num < 0 || num > 1)) {
      alert("Please enter exactly three decimal numbers between 0 and 1, separated by spaces.");
      return;
    }

    try {
      const response = await fetch('http://localhost:5001/submit', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ decimals: decimals })
      });

      if (!response.ok) {
        throw new Error(`An error has occurred: ${response.status}`);
      }

      const result = await response.json();
      setResponse(result.response);
    } catch (error) {
      console.error('Error:', error);
      setResponse('Failed to send data to Flask');
    }
  };

  useEffect(() => {
    let isMounted = true;
    const fetchDisplayData = async () => {
      try {
        const response = await fetch('http://localhost:5001/display');
        if (!response.ok) {
          throw new Error('Failed to fetch display data');
        }
        const data = await response.json();
        setDisplayData(data);
      } catch (error) {
        console.error('Error fetching display data:', error);
        setDisplayData({ error: "Failed to fetch data" });
      }
    };

    const fetchResidualPlot = async () => {
      setResidualPlotLoading(true);
      try {
        const response = await fetch('http://localhost:5001/residual_plot');
        if (!response.ok) {
          throw new Error('Failed to fetch residual plot');
        }
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        if (isMounted) {
          setResidualPlot(url);
        }
      } catch (error) {
        console.error('Error fetching residual plot:', error);
      } finally {
        if (isMounted) {
          setResidualPlotLoading(false);
        }
      }
    };
  
    fetchResidualPlot();
    fetchDisplayData();
    return () => {
      isMounted = false;
      URL.revokeObjectURL(residualPlot); // Clean up blob URL
    };
  }, []); 
  return (
    <div className="App">
      <header className="App-header">
        <div className="display-box" style={{ padding: '20px', margin: '20px', border: '2px solid #ccc', fontSize: '14px' }}>
          <h2>Display Data</h2>
          {displayData ? (
            Object.entries(displayData).map(([key, value]) => (
              <div key={key}>
                <strong>{key}</strong>: {JSON.stringify(value, null, 2)}
              </div>
            ))
          ) : (
            <p>Loading display data...</p>
          )}
        </div>
        <h1>Generate Plot</h1>
      <input
        type="text"
        placeholder="Enter your question..."
        value={question}
        onChange={handleQuestionChange}
      />
      <button onClick={handleGenerateCode} disabled={loading}>
        {loading ? 'Loading...' : 'Generate Code and Plot'}
      </button>
      {responseCode && (
        <div>
          <h2>Generated Python Code</h2>
          <pre className="code-block">
              <code>{responseCode}</code>
          </pre>
        </div>
      )}
      {PlotUrl && (
        <div>
          <h2>Plot</h2>
          <img src={PlotUrl} alt="Plot required by the user" style={{ maxWidth: '100%' }} />
        </div>
      )}

      <h1>Chat with AI</h1>
      <div className="chat-box" style={{ padding: '20px', margin: '20px', border: '2px solid #ccc', fontSize: '14px', height: '400px', overflowY: 'scroll', width: '100%', maxWidth: '600px' }}>
      {chatHistory.map((chat, index) => (
            <div key={index} style={{ marginBottom: '10px' }}>
              <strong>Question:</strong> {chat.question}<br />
              <strong>Answer:</strong> {chat.answer}
            </div>
          ))}
        </div>
        <input
          type="text"
          placeholder="Type your question here..."
          value={inputValue}
          onChange={handleInputChange}
        />
        <button onClick={handleSubmit}>Submit</button>
        
        <h2>What do you think the feature weights are?</h2>
        <input
          type="text"
          placeholder="Example: 0.3 0.5 0.7"
          value={decimalInput}
          onChange={handleDecimalInputChange}
        />
        <button onClick={handleDecimalSubmit}>Submit Decimals</button>

        <div className="display-box" style={{ padding: '10px', margin: '100px', border: '7px solid #ccc', fontSize: '20px', backgroundColor: 'white', color: 'grey' }}>
          Response: {response}
        </div>
      </header>
    </div>
  );
}

export default App; 
