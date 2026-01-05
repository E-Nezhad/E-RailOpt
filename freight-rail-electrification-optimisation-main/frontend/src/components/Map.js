import React from "react";
import { MapContainer, TileLayer, Marker, Popup, Polyline } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

// Utility to create emoji icon
const createEmojiIcon = (emoji) =>
  L.divIcon({
    className: "emoji-marker",
    html: `<div style="font-size: 28px;">${emoji}</div>`,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
  });

const startIcon = createEmojiIcon("🚦");
const endIcon = createEmojiIcon("🏁");
const stationIcon = createEmojiIcon("🚉");
const deployedStationIcon = createEmojiIcon("🟢"); // or any emoji/icon you like


const MapComponent = ({ railwaySegments = [], startPos, endPos, freightPath = [] }) => {
  // Calculate map center based on start and end positions
  const calculateCenter = () => {
    if (
      Array.isArray(startPos) &&
      Array.isArray(endPos) &&
      startPos.length === 2 &&
      endPos.length === 2
    ) {
      return [(startPos[0] + endPos[0]) / 2, (startPos[1] + endPos[1]) / 2];
    } else if (Array.isArray(startPos) && startPos.length === 2) {
      return startPos;
    } else if (Array.isArray(endPos) && endPos.length === 2) {
      return endPos;
    }
    return [-33.86, 151.20];
  };

  const getPositionFromChainage = (chainage, path) => {
    if (!path || path.length < 2) return null;

    let accumulatedDistance = 0;

    for (let i = 0; i < path.length - 1; i++) {
      const pointA = L.latLng(path[i]);
      const pointB = L.latLng(path[i + 1]);
      const segmentDistance = pointA.distanceTo(pointB); // meters

      if (accumulatedDistance + segmentDistance >= chainage) {
        // The target point is somewhere on this segment
        const remaining = chainage - accumulatedDistance;
        const ratio = remaining / segmentDistance;

        const lat = pointA.lat + (pointB.lat - pointA.lat) * ratio;
        const lng = pointA.lng + (pointB.lng - pointA.lng) * ratio;

        return [lat, lng];
      }

      accumulatedDistance += segmentDistance;
    }

    // If we didn't reach the chainage (e.g. too short path), return last point
    return path[path.length - 1];
  };

  const getStationMarkersFromParsedResults = (path) => {
  try {
    const parsed = JSON.parse(localStorage.getItem("parsedResults"));
    if (!parsed || !parsed.schedule_results) return [];

    const schedule = parsed.schedule_results;
    const entries = Object.entries(schedule);

    // Filter out origin/destination if you only want intermediate stations
    // const stationEntries = entries.filter(([name]) => name !== "origin" && name !== "destination");
    const stationEntries = entries

    const maxChainage = Math.max(...stationEntries.map(([_, v]) => v.chainage || 0));

    return stationEntries.map(([name, station]) => {
      const pos = getPositionFromChainage(station.chainage, path, maxChainage);
      if (!pos) return null;

      return {
        name,
        chainage: station.chainage,
        position: pos,
        arrival: station.arrival_time,
        departure: station.departure_time,
        delay: station.delay,
        deployed: station.deployed || false,  // Add this line
      };
    }).filter(Boolean);
  } catch (e) {
    console.error("Error parsing 'parsedResults' from localStorage:", e);
    return [];
  }
};


  const stationMarkers = getStationMarkersFromParsedResults(freightPath);

  return (
    <div
      style={{
        border: "2px solid #ddd",
        borderRadius: "8px",
        padding: "10px",
        boxShadow: "0 4px 8px rgba(0,0,0,0.1)",
      }}
    >
      <MapContainer
        center={calculateCenter()}
        zoom={10}
        style={{ height: "400px", width: "100%", borderRadius: "8px" }}
      >
        <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />

        {Array.isArray(startPos) && (
          <Marker position={startPos} icon={startIcon}>
            <Popup>Start Point</Popup>
          </Marker>
        )}

        {Array.isArray(endPos) && (
          <Marker position={endPos} icon={endIcon}>
            <Popup>End Point</Popup>
          </Marker>
        )}

        {railwaySegments.map((segment, idx) => (
          <Polyline
            key={idx}
            positions={segment}
            color="blue"
            weight={4}
            opacity={0.7}
          />
        ))}

        {freightPath && freightPath.length > 1 && (
          <Polyline
            positions={freightPath}
            color="purple"
            weight={5}
            opacity={0.9}
          />
        )}

        {stationMarkers.map((station, idx) => (
          <Marker
            key={`station-${idx}`}
            position={station.position}
            icon={station.deployed ? deployedStationIcon : stationIcon}
          >
            <Popup>
              <div>
                <strong>{station.name}</strong><br />
                Chainage: {station.chainage}m<br />
                Arrival: {station.arrival}<br />
                Departure: {station.departure}<br />
                Delay: {station.delay}
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
};

export default MapComponent;
