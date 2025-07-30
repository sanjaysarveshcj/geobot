import React, { useEffect } from "react";
import {
  MapContainer,
  TileLayer,
  Circle,
  Marker,
  Popup,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { renderToStaticMarkup } from "react-dom/server";
import LocationOnIcon from "@mui/icons-material/LocationOn";

// ✅ Fit map bounds to all marker positions
const FitBoundsToDisasters = ({ positions }) => {
  const map = useMap();

  useEffect(() => {
    if (positions.length > 0) {
      const bounds = L.latLngBounds(positions);
      map.flyToBounds(bounds, {
        padding: [50, 50],
        duration: 1.5,
        maxZoom: 15, // ✅ Prevents zooming in too much
      });
    }
  }, [positions, map]);

  return null;
};
// ✅ Custom offset for overlapping markers (currently no offset)
const applyOffset = (lat, lng, index, name) => {
  const offset = 0;
  return { lat: lat + offset, lng: lng + offset, name };
};

// ✅ Create MUI icon for Leaflet
const createMuiIcon = (color = "red") =>
  L.divIcon({
    html: renderToStaticMarkup(
      <LocationOnIcon style={{ color, fontSize: "32px" }} />
    ),
    className: "",
    iconSize: [32, 32],
    iconAnchor: [16, 32],
  });

const DisasterMap = ({ disasterData }) => {
  const defaultCenter = [23.5, 80.5];

  // ✅ Default fallback data
  const defaultData = [
    {
      name: "Bangalore",
      latitude: 12.9716,
      longitude: 77.5946,
      total_affected: 50000,
      color: "orange",
    },
    {
      name: "Hyderabad",
      latitude: 17.385,
      longitude: 78.4867,
      total_affected: 30000,
      color: "yellow",
    },
  ];

  // ✅ Use real data if valid
  const activeData =
    Array.isArray(disasterData) && disasterData.length > 0
      ? disasterData
      : defaultData;

  // ✅ Disaster zones with popup info
  const disasterZones = activeData.map((item, index) => ({
    id: `${item.name}-${index}`,
    name: item.name || `Zone-${index}`,
    position: [item.latitude, item.longitude],
    radius: 800,
    color: item.color || "red",
    affected: item.total_affected ?? 0,
  }));

  // ✅ User marker locations (used for blue icons)
  const userLocations = activeData.map((item, index) =>
    applyOffset(item.latitude, item.longitude, index, item.name)
  );

  return (
    <div className="w-full md:w-3/4 h-[95vh] p-4 bg-gray-100">
      <h2 className="text-2xl font-bold mb-4 text-red-700">🗺️ Disaster Map</h2>
      <MapContainer
        center={defaultCenter}
        zoom={5}
        style={{ height: "95%", width: "100%" }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* ✅ Dynamically adjust map view to fit all zones */}
        <FitBoundsToDisasters positions={disasterZones.map((z) => z.position)} />

        {/* ✅ Render disaster zones with markers and circles */}
        {disasterZones.map((zone, index) => (
          <React.Fragment key={`zone-${index}`}>
            <Marker position={zone.position} icon={createMuiIcon("red")}>
              <Popup>
                🚨 {zone.name} <br />
                Affected: {zone.affected.toLocaleString()}
              </Popup>
            </Marker>
            <Circle
              center={zone.position}
              radius={zone.radius}
              pathOptions={{ color: zone.color, fillOpacity: 0.4 }}
            />
          </React.Fragment>
        ))}

        {/* ✅ Render user locations with blue MUI markers */}
        {userLocations.map((loc, idx) => (
          <Marker
            key={`user-${idx}`}
            position={[loc.lat, loc.lng]}
            icon={createMuiIcon("blue")}
          >
            <Popup>📍 {loc.name}</Popup>
          </Marker>
        ))}
      </MapContainer>
    </div>
  );
};

export default DisasterMap;
