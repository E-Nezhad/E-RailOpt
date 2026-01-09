import React, { useState, useEffect } from "react";
import MapComponent from "./Map"; // Your existing Map component
import SuggestedStops from "./suggestedStops"; // Assuming this exports an array or component
import MapWithStations from "./MapWithStations";

const RailwayExtractor = () => {
  // State with localStorage initialization
  const [startCoordinates, setStartCoordinates] = useState(() => localStorage.getItem("startCoordinates") || "");
  const [endCoordinates, setEndCoordinates] = useState(() => localStorage.getItem("endCoordinates") || "");
  const [startLat, setStartLat] = useState(() => localStorage.getItem("startLat") || "");
  const [startLon, setStartLon] = useState(() => localStorage.getItem("startLon") || "");
  const [endLat, setEndLat] = useState(() => localStorage.getItem("endLat") || "");
  const [endLon, setEndLon] = useState(() => localStorage.getItem("endLon") || "");
  const [railwaySegments, setRailwaySegments] = useState(() => {
    const saved = localStorage.getItem("railwaySegments");
    return saved ? JSON.parse(saved) : [];
  });
  const [freightPath, setFreightPath] = useState(() => {
    const saved = localStorage.getItem("freightPath");
    return saved ? JSON.parse(saved) : [];
  });

  // You will replace this with your actual recommended stops data, or fetch it as needed
  const [suggestedStops, setSuggestedStops] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showStops, setShowStops] = useState(false);
  

  // Save state to localStorage on changes
  useEffect(() => {
    localStorage.setItem("startCoordinates", startCoordinates);
    localStorage.setItem("endCoordinates", endCoordinates);
    localStorage.setItem("startLat", startLat);
    localStorage.setItem("startLon", startLon);
    localStorage.setItem("endLat", endLat);
    localStorage.setItem("endLon", endLon);
  }, [startCoordinates, endCoordinates, startLat, startLon, endLat, endLon]);

  useEffect(() => {
    localStorage.setItem("railwaySegments", JSON.stringify(railwaySegments));
  }, [railwaySegments]);

  useEffect(() => {
    localStorage.setItem("freightPath", JSON.stringify(freightPath));
  }, [freightPath]);

  const handleStartCoordinatesChange = (event) => {
    const value = event.target.value;
    setStartCoordinates(value);
    const coords = value.split(",").map(coord => coord.trim());
    if (coords.length === 2) {
      setStartLat(coords[0]);
      setStartLon(coords[1]);
    } else {
      setStartLat("");
      setStartLon("");
    }
  };

  const handleEndCoordinatesChange = (event) => {
    const value = event.target.value;
    setEndCoordinates(value);
    const coords = value.split(",").map(coord => coord.trim());
    if (coords.length === 2) {
      setEndLat(coords[0]);
      setEndLon(coords[1]);
    } else {
      setEndLat("");
      setEndLon("");
    }
  };

  const handleFetchRailways = async () => {
    setError("");
    if (!startLat || !startLon || !endLat || !endLon) {
      setError("Please enter all coordinates.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:5002/api/get-railways", {
        // add modal here. 
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          startLat: parseFloat(startLat),
          startLon: parseFloat(startLon),
          endLat: parseFloat(endLat),
          endLon: parseFloat(endLon),
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to fetch railway data");

      setRailwaySegments(data.segments);
      await handleFetchFreightPath();
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFetchFreightPath = async () => {
    try {
      // probably should add a modal here. 
      const res = await fetch("http://127.0.0.1:5002/api/get-railway-path", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          startLat: parseFloat(startLat),
          startLon: parseFloat(startLon),
          endLat: parseFloat(endLat),
          endLon: parseFloat(endLon),
        }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to fetch freight path");

      setFreightPath(data.path);

      // Example: simulate fetching suggested stops here or integrate your actual data
      // setSuggestedStops(data.suggestedStops || []);
    } catch (e) {
      setError(e.message);
    }
  };

  const handleReset = () => {
    setStartCoordinates("");
    setEndCoordinates("");
    setStartLat("");
    setStartLon("");
    setEndLat("");
    setEndLon("");
    setRailwaySegments([]);
    setFreightPath([]);
    setSuggestedStops([]);
    setError("");

    localStorage.removeItem("startCoordinates");
    localStorage.removeItem("endCoordinates");
    localStorage.removeItem("startLat");
    localStorage.removeItem("startLon");
    localStorage.removeItem("endLat");
    localStorage.removeItem("endLon");
    localStorage.removeItem("railwaySegments");
    localStorage.removeItem("freightPath");
  };

  const center =
    startLat && startLon && endLat && endLon
      ? [
          (parseFloat(startLat) + parseFloat(endLat)) / 2,
          (parseFloat(startLon) + parseFloat(endLon)) / 2,
        ]
      : [-33.86, 151.20];

  const startPos = startLat && startLon ? [parseFloat(startLat), parseFloat(startLon)] : null;
  const endPos = endLat && endLon ? [parseFloat(endLat), parseFloat(endLon)] : null;

    const fetchRecommendedStops = () => {
        if (!freightPath || freightPath.length === 0) return [];
        const count = 3;
        const interval = Math.floor(freightPath.length / (count + 1));
        const stops = [];
        for (let i = 1; i <= count; i++) {
            stops.push(freightPath[i * interval]);
        }
        return stops;
    };

    const toggleShowStops = () => {
    if (!showStops) {
        const stops = fetchRecommendedStops();
        setSuggestedStops(stops);
    } else {
        setSuggestedStops([]);
    }
    setShowStops(!showStops);
    };


  return (
    <div>
      <h2>Freight Railway Route Extractor</h2>

      <div style={{ marginBottom: "10px" }}>
        <label style={{ display: "block", marginBottom: "4px", fontWeight: "bold" }}>
          Start Coordinates (Lat, Lon):
        </label>
        <input
          type="text"
          value={startCoordinates}
          onChange={handleStartCoordinatesChange}
          placeholder="e.g., -33.865, 151.209"
          style={{
            width: "100%",
            padding: "8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            boxSizing: "border-box"
          }}
        />
      </div>

      <div style={{ marginBottom: "10px" }}>
        <label style={{ display: "block", marginBottom: "4px", fontWeight: "bold" }}>
          End Coordinates (Lat, Lon):
        </label>
        <input
          type="text"
          value={endCoordinates}
          onChange={handleEndCoordinatesChange}
          placeholder="e.g., -33.920, 151.234"
          style={{
            width: "100%",
            padding: "8px",
            borderRadius: "4px",
            border: "1px solid #ccc",
            boxSizing: "border-box"
          }}
        />
      </div>

      <div style={{ display: "flex", gap: "12px", marginTop: "20px" }}>
        <button
          onClick={handleFetchRailways}
          disabled={loading}
          style={{
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            padding: "10px 20px",
            borderRadius: "6px",
            fontSize: "16px",
            fontWeight: "500",
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.6 : 1,
            transition: "background-color 0.3s ease",
          }}
        >
          {loading ? "Loading..." : "Get Freight Railways & Suggested Stops"}
        </button>

        {/* <button
          onClick={toggleShowStops}
          disabled={freightPath.length === 0}
          style={{
            backgroundColor: showStops ? "#28a745" : "#17a2b8",
            color: "white",
            border: "none",
            padding: "10px 20px",
            borderRadius: "6px",
            fontSize: "16px",
            fontWeight: "500",
            cursor: freightPath.length === 0 ? "not-allowed" : "pointer",
            opacity: freightPath.length === 0 ? 0.5 : 1,
            transition: "background-color 0.3s ease",
          }}
        >
          {showStops ? "Hide Recommended Stops" : "Show Recommended Stops"}
        </button> */}

        <button
          onClick={handleReset}
          style={{
            backgroundColor: "#dc3545",
            color: "white",
            border: "none",
            padding: "10px 20px",
            borderRadius: "6px",
            fontSize: "16px",
            fontWeight: "500",
            cursor: "pointer",
            transition: "background-color 0.3s ease",
          }}
        >
          Reset All
        </button>
      </div>

        <MapComponent
            center={center}
            railwaySegments={railwaySegments}
            startPos={startPos}
            endPos={endPos}
            freightPath={freightPath}
            suggestedStops={showStops ? suggestedStops : []}  // Add this prop
        />

    </div>
  );
};

export default RailwayExtractor;
