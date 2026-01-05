import React, { useEffect, useState } from "react";
import MapComponent from "./Map";

const MapWithStations = ({ railwaySegments, startPos, endPos, freightPath, suggestedStops }) => {
  const [stations, setStations] = useState({});

  useEffect(() => {
    const stored = localStorage.getItem("parsedResults");
    console.log("stored")
    console.log(stored)
    if (stored) {
      try {
        const parsedObj = JSON.parse(stored);
        const stationData = parsedObj.schedule_results; // access the station object
        console.log("parsed", stationData);

        setStations(stationData); // ✅ this should now be correct
      } catch (err) {
        console.error("Failed to parse parsedResults from localStorage:", err);
      }
    }
  }, []);

  return (
    <MapComponent
      railwaySegments={railwaySegments}
      startPos={startPos}
      endPos={endPos}
      freightPath={freightPath}
      suggestedStops={suggestedStops}
      stations={stations} // 👈 inject station data
    />
  );
};

export default MapWithStations;
