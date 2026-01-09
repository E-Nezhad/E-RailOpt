import React, { useEffect, useState } from "react";

const ResultsTable = () => {
  const [results, setResults] = useState(null);

  useEffect(() => {
    const stored = localStorage.getItem("parsedResults");
    console.log("Loaded from localStorage:", stored);

    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        console.log("Parsed results object:", parsed);
        setResults(parsed);
      } catch (e) {
        console.error("Error parsing stored results:", e);
      }
    }
  }, []);

  if (!results) {
    return <p>No results found. Please upload a file first.</p>;
  }

  const cellStyle = {
    border: "1px solid #ddd",
    padding: "8px",
    textAlign: "left",
    verticalAlign: "top",
  };

  const headerStyle = {
    ...cellStyle,
    backgroundColor: "#f4f4f4",
    fontWeight: "bold",
  };

  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "14px" }}>
      <thead>
        <tr>
          <th style={headerStyle}>Station</th>
          <th style={headerStyle}>Chainage</th>
          <th style={headerStyle}>Arrival</th>
          <th style={headerStyle}>Departure</th>
          <th style={headerStyle}>Delay - Charging or Swapping Time (hrs)</th>
          <th style={headerStyle}>Deployed</th>
          <th style={headerStyle}>Container</th>
          <th style={headerStyle}>Action</th>
          <th style={headerStyle}>Arrival SOC</th>
          <th style={headerStyle}>Departure SOC</th>
        </tr>
      </thead>
      <tbody>
        {Object.values(results.schedule_results)
          .sort((a, b) => {
            if (a.station_index !== undefined && b.station_index !== undefined) {
              return a.station_index - b.station_index;
            }
            return a.chainage - b.chainage;
          })
          .map((station, index) => {
            if (!station) {
              return (
                <tr key={`null-${index}`}>
                  <td colSpan="10" style={{ color: "red", padding: "8px" }}>
                    Warning: station is null or undefined at index {index}
                  </td>
                </tr>
              );
            }

            const containerEntries = station.containers
              ? Object.entries(station.containers)
              : [];

            // No containers – just one row
            if (containerEntries.length === 0) {
              return (
                <tr key={index}>
                  <td style={cellStyle}>{station.name}</td>
                  <td style={cellStyle}>{station.chainage}</td>
                  <td style={cellStyle}>{station.arrival_time}</td>
                  <td style={cellStyle}>{station.departure_time}</td>
                  <td style={cellStyle}>{station.delay}</td>
                  <td style={cellStyle}>{station.deployed ? "Yes" : "No"}</td>
                  <td style={cellStyle} colSpan={4}>No containers</td>
                </tr>
              );
            }

            // If containers exist, return multiple rows
            return containerEntries.map(([cname, c], cidx) => {
              const cleanName = cname.split(" in ")[0];
              return (
                <tr key={`${index}-${cidx}`}>
                  {cidx === 0 ? (
                    <>
                      <td style={cellStyle} rowSpan={containerEntries.length}>{station.name}</td>
                      <td style={cellStyle} rowSpan={containerEntries.length}>{station.chainage}</td>
                      <td style={cellStyle} rowSpan={containerEntries.length}>{station.arrival_time}</td>
                      <td style={cellStyle} rowSpan={containerEntries.length}>{station.departure_time}</td>
                      <td style={cellStyle} rowSpan={containerEntries.length}>{station.delay}</td>
                      <td style={cellStyle} rowSpan={containerEntries.length}>{station.deployed ? "Yes" : "No"}</td>
                    </>
                  ) : null}
                  <td style={cellStyle}>{cleanName}</td>
                  <td style={cellStyle}>{c.action}</td>
                  <td style={cellStyle}>{c.arrival_soc ?? "-"}%</td>
                  <td style={cellStyle}>{c.departure_soc ?? "-"}%</td>
                </tr>
              );
            });
          })}
      </tbody>
    </table>
  );
};

export default ResultsTable;
