import React, { useState } from "react";
import DisasterChat from "./components/DisasterChat";
import DisasterMap from "./components/DisasterMap";


function App() {
  const [disasterData, setDisasterData] = useState(null);
  return (
    <div className="flex flex-col md:flex-row h-screen p-4 gap-4 bg-gray-50">
      <DisasterChat setDisasterData={setDisasterData} />
      <DisasterMap disasterData={disasterData} />
    </div>
  );
}

export default App;
