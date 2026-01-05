import React, { useEffect, useState } from "react";
import * as XLSX from "xlsx";
import "./MatrixViewer.css"; 

const MatrixViewer = () => {
  const [matrices, setMatrices] = useState({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [statusMessage, setStatusMessage] = useState("");

  useEffect(() => {
    const fetchExcel = async () => {
      try {
        const response = await fetch("http://127.0.0.1:5002/uploads/energy_matrix_ui.xlsx");
        if (!response.ok) throw new Error("No matrix data found. Please upload train data.");

        const arrayBuffer = await response.arrayBuffer();
        const workbook = XLSX.read(arrayBuffer, { type: "buffer" });

        const matrixData = {};
        workbook.SheetNames.forEach((name) => {
          const sheet = workbook.Sheets[name];
          let data = XLSX.utils.sheet_to_json(sheet, { header: 1 });

          if (name === "Energy Matrix") {
            data[0][0] = "Energy Consumption (kWh)";
          } else if (name === "Time Matrix") {
            data[0][0] = "Time (hrs)";

            // Convert all non-header cells to hours (from seconds)
            data = data.map((row, i) =>
              row.map((cell, j) => {
                if (i === 0 || j === 0) return cell; // skip row/col headers
                const num = parseFloat(cell);
                return !isNaN(num) ? (num / 3600).toFixed(2) : cell;
              })
            );
          }

          matrixData[name] = data;
        });

        setMatrices(matrixData);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchExcel();
  }, []);


  const runAlgorithm = async () => {
    try {
      setStatusMessage("Running optimization algorithm...");
      const response = await fetch("http://127.0.0.1:5002/run-algorithm", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        // If needed: send any data required by the backend
        body: JSON.stringify({ trigger: true })
      });

      if (!response.ok) throw new Error("Failed to run algorithm.");
      const result = await response.json();
      setStatusMessage(result.message || "Algorithm ran successfully.");
    } catch (err) {
      setStatusMessage(err.message);
    }
  };

  if (loading) return <p>Loading matrix...</p>;
  if (error) return <p style={{ color: "red" }}>{error}</p>;

  return (
    <div className="matrix-container">
      <h2>Energy and Time Matrices</h2>
      <button onClick={runAlgorithm} className="run-algo-btn">
        Run Optimization Algorithm
      </button>
      {statusMessage && <p>{statusMessage}</p>}
      {Object.entries(matrices).map(([name, matrix]) => (
        <div key={name} className="matrix-block">
          <h3>{name}</h3>
          <div className="matrix-scroll">
            <table className="matrix-table">
                <thead>
                    <tr>
                      <th>
                        {name.toLowerCase().includes("energy") && "Energy Consumption (kWh)"}
                        {name.toLowerCase().includes("time") && "Time (hrs)"}
                      </th>
                    {matrix[0]?.slice(1).map((col, colIndex) => (
                        <th key={colIndex}>{col}</th>
                    ))}
                    </tr>
                </thead>
                <tbody>
                    {matrix.slice(1).map((row, rowIndex) => (
                    <tr key={rowIndex}>
                        <th>{row[0]}</th> {/* Row header */}
                        {row.slice(1).map((cell, colIndex) => (
                        <td key={colIndex}>{cell}</td>
                        ))}
                    </tr>
                    ))}
                </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
};

export default MatrixViewer;
