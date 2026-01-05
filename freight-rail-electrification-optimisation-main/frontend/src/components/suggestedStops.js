import { Marker, Popup } from "react-leaflet";
import L from "leaflet";

const stopIcon = L.divIcon({
  className: "stop-marker",
  html: `<div style="font-size: 24px;">🛑</div>`,
  iconSize: [24, 24],
  iconAnchor: [12, 12],
});

const SuggestedStops = ({ stopPoints = [] }) => {
  return (
    <>
      {stopPoints.map((pos, idx) => (
        <Marker key={idx} position={pos} icon={stopIcon}>
          <Popup>Suggested Stop #{idx + 1}</Popup>
        </Marker>
      ))}
    </>
  );
};

export default SuggestedStops;
