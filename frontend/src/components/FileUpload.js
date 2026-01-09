import React, { useState, useEffect } from "react";
import { Button, InputGroup, FormControl, ListGroup, Alert, Row, Col, Modal, Spinner } from 'react-bootstrap';
import * as XLSX from 'xlsx';

const FileUpload = () => {
  const assumptionLabels = {
    P: "Power (W)",
    m_loc: "Mass of Locomotive (kg)",
    m_car: "Mass of Car (kg)",
    K_cl: "Constant K Cl",
    num_locos: "Number of Locos",
    num_cars: "Number of Cars",
    axles_on_loco: "Axles on Loco",
    axles_on_car: "Axles on Car",
    fa_cl: "Frontal Area (m^2)",
    train_efficiency: "Train Efficiency",
    regen_efficiency: "Regenerative Efficiency",
    regen_buffer: "Regen Buffer",
    max_decel: "Maximum Deceleration (m/s^2)",
    brake_buffer: "Brake Buffer",
    max_tractive_force: "Maximum Tractive Force (N)",
    max_accel: "Maximum Acceleration (m/s^2)"
  };

  const defaultAssumptions = {
    P: 3000 * 1000,
    m_loc: 150 * 1000,
    m_car: 100 * 1000,
    K_cl: 0.0055,
    num_locos: 4,
    num_cars: 7,
    axles_on_loco: 4,
    axles_on_car: 4,
    fa_cl: 10,
    train_efficiency: 0.85,
    regen_efficiency: 0.5,
    regen_buffer: 0.0,
    max_decel: 0.5,
    brake_buffer: 30,
    max_tractive_force: 300000,
    max_accel: 1.0,
  };

  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const [stationFile, setStationFile] = useState(null);
  const [isStationFileUploaded, setIsStationFileUploaded] = useState(false);
  const [assumptions, setAssumptions] = useState(() => JSON.parse(localStorage.getItem('assumptions')) || { ...defaultAssumptions });
  const [stations, setStations] = useState(() => JSON.parse(localStorage.getItem('stations')) || []);
  const [currentStation, setCurrentStation] = useState("");
  const [isUploaded, setIsUploaded] = useState(() => localStorage.getItem('isUploaded') === 'true');
  const [matrixGenerated, setMatrixGenerated] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [stationDetails, setStationDetails] = useState(() => {const stored = localStorage.getItem('stationDetails');return stored ? JSON.parse(stored) : [];});
  const [showOptimisationModal, setShowOptimisationModal] = useState(false);
  const [showResultsModal, setShowResultsModal] = useState(false);



  useEffect(() => {
    localStorage.setItem('assumptions', JSON.stringify(assumptions));
  }, [assumptions]);

  useEffect(() => {
    localStorage.setItem('stations', JSON.stringify(stations));
  }, [stations]);

  useEffect(() => {
    localStorage.setItem('isUploaded', isUploaded);
  }, [isUploaded]);

  useEffect(() => {
    localStorage.setItem('matrixGenerated', matrixGenerated);
  }, [matrixGenerated]);

  useEffect(() => {
  localStorage.setItem('stationDetails', JSON.stringify(stationDetails));
  }, [stationDetails]);


  const handleReset = async () => {
    setAssumptions({ ...defaultAssumptions });
    setStations([]);
    setStationDetails([]);
    setFile(null);
    setIsUploaded(false);
    setMatrixGenerated(false);
    setError('');
    localStorage.clear();
    try {
      await fetch("http://127.0.0.1:5002/reset", { method: "DELETE" });
    } catch (err) {
      console.error("Failed to delete file:", err);
    }
  };

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
    setIsUploaded(false);
  };

  const handleInputChange = (event) => {
    setAssumptions({ ...assumptions, [event.target.name]: parseFloat(event.target.value) });
  };

  const handleStationChange = (event) => {
    setCurrentStation(event.target.value);
  };

  const handleAddStations = () => {
    if (currentStation) {
      const stationArray = currentStation.split(",").map(s => s.trim()).filter(s => !isNaN(s) && s !== "");
      const numericStations = stationArray.map(parseFloat);
      if (numericStations.length > 0) {
        const newStations = [...stations, ...numericStations];
        setStations(newStations);

        const newDetails = numericStations.map(station => ({
          station,
          setupCost: "",
          maxCharging: "",
          maxBatteries: ""
        }));
        setStationDetails([...stationDetails, ...newDetails]);
        setCurrentStation("");
      } else {
        setError("Please enter valid station values separated by commas.");
      }
    }
  };

  const handleDetailChange = (index, field, value) => {
    const updatedDetails = [...stationDetails];
    updatedDetails[index][field] = value;
    setStationDetails(updatedDetails);
  };


  const handleRemoveStation = (index) => {
    setStations(stations.filter((_, i) => i !== index));
  };

  const handleUpload = async () => {
    if (!file) {
      setError('No file selected!');
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:5002/upload", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("File upload failed");
      alert('File uploaded successfully');
      setIsUploaded(true);
    } catch (error) {
      setError(error.message);
    }
  };

  const handleRunEnergyMatrix = async () => {
    setLoading(true); // Start loading
    setShowModal(true);
    setError('');
    try {
      if (!isUploaded) throw new Error("Please upload the Excel file first.");
      const response = await fetch("http://127.0.0.1:5002/generate-energy-matrix", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ assumptions, stations }),
      });
      if (!response.ok) throw new Error("Energy matrix generation failed");
      await response.json();
      setMatrixGenerated(true);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false); // End loading
      setShowModal(false);

    }
  };

  const handleRunOptimisation = async () => {
    setShowOptimisationModal(true); // Show modal
    try {
      console.log("Sending stationDetails:", stationDetails);  // <== Add this to debug
      const response = await fetch("http://127.0.0.1:5002/run-optimisation", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ assumptions, stations, stationDetails }),
      });

      if (!response.ok) {
        throw new Error("Optimisation failed. Please check input values.");
      }

      const result = await response.json();
      alert("Optimisation complete! Output saved as Excel.");
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      setShowOptimisationModal(false); // Hide modal after completion
    }
  };

  const handleResults = async () => {
    //setShowResultsModal(true); // Show modal
    try {
      const response = await fetch("http://127.0.0.1:5002/parse-results", {
        method: "POST",
        headers: {
        "Content-Type": "application/json"
        },
        body: JSON.stringify({ stations })  // Send the stations array here
      
      });

      if (!response.ok) {
        throw new Error("Failed to parse results.");
      }

      const result = await response.json();
      console.log("Parsed result:", result); // Optional: log or store the parsed data
      
      // Save result to localStorage
      localStorage.setItem("parsedResults", JSON.stringify(result));
      
      alert("Results successfully parsed!");
    } catch (err) {
      alert(`Error: ${err.message}`);
    } finally {
      //setShowOptimisationModal(false); // Hide modal after completion
    }
  };

  const handleStationFileUpload = () => {
    if (!stationFile) {
      setError("Please select a station data file.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const data = new Uint8Array(e.target.result);
      const workbook = XLSX.read(data, { type: "array" });

      const sheet = workbook.Sheets[workbook.SheetNames[0]];
      const jsonData = XLSX.utils.sheet_to_json(sheet);

      const extractedStations = [];
      const extractedDetails = [];

      jsonData.forEach((row) => {
        if (row.Chainage !== undefined && !isNaN(parseFloat(row.Chainage))) {
          const chainage = parseFloat(row.Chainage);
          extractedStations.push(chainage);
          extractedDetails.push({
            station: chainage,
            setupCost: row["Fixed Set Up Cost"] || 0,
            maxCharging: row["Max Charging"] || 0,
            maxBatteries: row["Max Batteries"] || 0,
            nextStation: row["Next Station"] // Optional: Could be filled manually or computed
          });
        }
      });

      setStations((prev) => [...prev, ...extractedStations]);
      setStationDetails((prev) => [...prev, ...extractedDetails]);
      setIsStationFileUploaded(true);
    };

    reader.readAsArrayBuffer(stationFile);
  };



  return (
    <div className="p-3">
      <Row>
        <Col md={8}><h3>Input Train Data</h3></Col>
        <Col md={4}><h3>Upload Route Data</h3></Col>
      </Row>
      <Row>
        <Col md={4}>
          {Object.keys(assumptions).filter(key => !['fa_cl','train_efficiency','regen_efficiency','regen_buffer','max_decel','brake_buffer','max_tractive_force','max_accel'].includes(key)).map(key => (
            <InputGroup className="mb-2" key={key}>
              <InputGroup.Text style={{ width: '250px', textAlign: 'left' }}>{assumptionLabels[key]}</InputGroup.Text>
              <FormControl type="number" name={key} value={assumptions[key]} onChange={handleInputChange} style={{ flex: 1, minHeight: '35px' }} />
            </InputGroup>
          ))}
        </Col>
        <Col md={4}>
          {Object.keys(assumptions).filter(key => ['fa_cl','train_efficiency','regen_efficiency','regen_buffer','max_decel','brake_buffer','max_tractive_force','max_accel'].includes(key)).map(key => (
            <InputGroup className="mb-2" key={key}>
              <InputGroup.Text style={{ width: '250px', textAlign: 'left' }}>{assumptionLabels[key]}</InputGroup.Text>
              <FormControl type="number" name={key} value={assumptions[key]} onChange={handleInputChange} style={{ flex: 1, minHeight: '35px' }} />
            </InputGroup>
          ))}
        </Col>
        <Col md={4}>
          <InputGroup className="mb-3">
            <FormControl type="file" onChange={handleFileChange} />
            <Button variant="outline-secondary" onClick={handleUpload}>Upload Route Data</Button>
          </InputGroup>
          <h3>Upload Station Data</h3>
          <InputGroup className="mb-3">
            <FormControl type="file" onChange={(e) => setStationFile(e.target.files[0])} />
            <Button variant="outline-secondary" onClick={handleStationFileUpload}>Upload Station Data</Button>
          </InputGroup>

        </Col>
      </Row>
      <Row className="mt-4">
        <Col md={12}>
          {isUploaded && (
            <>
              <h3>Enter Chainage Points (Stations) in meters</h3>
              <InputGroup className="mb-2">
                <FormControl type="text" value={currentStation} onChange={handleStationChange} placeholder="Enter stations separated by commas (e.g., 1000, 2000, 3000)" />
                <Button variant="outline-primary" onClick={handleAddStations}>Add Stations</Button>
              </InputGroup>
              <ListGroup className="mb-3">
                {stations.map((station, index) => (
                  <ListGroup.Item key={index} className="d-flex justify-content-between align-items-center">
                    {station} meters
                    <Button variant="danger" size="sm" onClick={() => handleRemoveStation(index)}>Remove</Button>
                  </ListGroup.Item>
                ))}
              </ListGroup>

              {stationDetails.length > 0 && (
                <div className="mt-4">
                  <h4>Station Parameters</h4>
                  <table className="table table-bordered">
                    <thead>
                      <tr>
                        <th>Station (m)</th>
                        <th>Fixed Setup Cost</th>
                        <th>Next Station</th>
                        <th>Max Charging</th>
                        <th>Max Batteries</th>
                      </tr>
                    </thead>
                    <tbody>
                      {stationDetails.map((detail, index) => (
                        <tr key={index}>
                          <td>{detail.station}</td>
                          <td>
                            <input
                              type="number"
                              className="form-control"
                              value={detail.setupCost}
                              onChange={(e) => handleDetailChange(index, 'setupCost', e.target.value)}
                            />
                          </td>
                          <td>
                            <input
                              type="string"
                              className="form-control"
                              value={detail.nextStation}
                              onChange={(e) => handleDetailChange(index, 'nextStation', e.target.value)}
                            />
                          </td>
                          <td>
                            <input
                              type="number"
                              className="form-control"
                              value={detail.maxCharging}
                              onChange={(e) => handleDetailChange(index, 'maxCharging', e.target.value)}
                            />
                          </td>
                          <td>
                            <input
                              type="number"
                              className="form-control"
                              value={detail.maxBatteries}
                              onChange={(e) => handleDetailChange(index, 'maxBatteries', e.target.value)}
                            />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}


              <Button onClick={handleRunEnergyMatrix} variant="primary" className="mt-2">Generate Energy Matrix</Button>{' '}
              <Button variant="success" onClick={handleRunOptimisation} className="mt-2">Run Optimisation</Button>{' '}
              <Modal show={showOptimisationModal} centered backdrop="static">
                <Modal.Body className="text-center">
                  <Spinner animation="border" role="status" className="mb-3" />
                  <div>Running optimisation... Please wait.</div>
                </Modal.Body>
              </Modal>

              <Button onClick={handleResults} variant="info" className="mt-2">
                Parse Results
              </Button>{' '}

              <Modal show={showResultsModal} backdrop="static" keyboard={false} centered>
                <Modal.Body className="d-flex flex-column align-items-center">
                  <Spinner animation="border" variant="primary" />
                  <div className="mt-3">Parsing Results. Please wait...</div>
                </Modal.Body>
              </Modal>

              <Button onClick={handleReset} variant="warning" className="mt-2">Reset</Button>
              <Modal show={showModal} backdrop="static" keyboard={false} centered>
                <Modal.Body className="d-flex flex-column align-items-center">
                  <Spinner animation="border" variant="primary" />
                  <div className="mt-3">Generating Energy Matrix. Please wait...</div>
                </Modal.Body>
              </Modal>

            </>
          )}
        </Col>
      </Row>
      {error && <Alert variant="danger" className="mt-3">{error}</Alert>}
      {matrixGenerated && <Alert variant="success" className="mt-3">Energy and Time Matrix successfully generated!</Alert>}
    </div>
  );
};

export default FileUpload;


// import React, { useState, useEffect } from "react";
// import { Button, InputGroup, FormControl, ListGroup, Alert, Row, Col, Modal, Spinner } from 'react-bootstrap';
// import * as XLSX from 'xlsx';

// const FileUpload = () => {
//   const assumptionLabels = {
//     P: "Power (W)",
//     m_loc: "Mass of Locomotive (kg)",
//     m_car: "Mass of Car (kg)",
//     K_cl: "Constant K Cl",
//     num_locos: "Number of Locos",
//     num_cars: "Number of Cars",
//     axles_on_loco: "Axles on Loco",
//     axles_on_car: "Axles on Car",
//     fa_cl: "Frontal Area (m^2)",
//     train_efficiency: "Train Efficiency",
//     regen_efficiency: "Regenerative Efficiency",
//     regen_buffer: "Regen Buffer",
//     max_decel: "Maximum Deceleration (m/s^2)",
//     brake_buffer: "Brake Buffer",
//     max_tractive_force: "Maximum Tractive Force (N)",
//     max_accel: "Maximum Acceleration (m/s^2)"
//   };

//   const defaultAssumptions = {
//     P: 3000 * 1000,
//     m_loc: 150 * 1000,
//     m_car: 100 * 1000,
//     K_cl: 0.0055,
//     num_locos: 4,
//     num_cars: 7,
//     axles_on_loco: 4,
//     axles_on_car: 4,
//     fa_cl: 10,
//     train_efficiency: 0.85,
//     regen_efficiency: 0.5,
//     regen_buffer: 0.0,
//     max_decel: 0.5,
//     brake_buffer: 30,
//     max_tractive_force: 300000,
//     max_accel: 1.0,
//   };

//   const [routeFile, setRouteFile] = useState(null);
//   const [stationFile, setStationFile] = useState(null);
//   const [error, setError] = useState('');
//   const [assumptions, setAssumptions] = useState(() => JSON.parse(localStorage.getItem('assumptions')) || { ...defaultAssumptions });
//   const [stations, setStations] = useState(() => JSON.parse(localStorage.getItem('stations')) || []);
//   const [currentStation, setCurrentStation] = useState("");
//   const [isUploaded, setIsUploaded] = useState(() => localStorage.getItem('isUploaded') === 'true');
//   const [matrixGenerated, setMatrixGenerated] = useState(false);
//   const [loading, setLoading] = useState(false);
//   const [showModal, setShowModal] = useState(false);
//   const [stationDetails, setStationDetails] = useState(() => { const stored = localStorage.getItem('stationDetails'); return stored ? JSON.parse(stored) : []; });
//   const [showOptimisationModal, setShowOptimisationModal] = useState(false);
//   const [showResultsModal, setShowResultsModal] = useState(false);

//   useEffect(() => {
//     localStorage.setItem('assumptions', JSON.stringify(assumptions));
//   }, [assumptions]);

//   useEffect(() => {
//     localStorage.setItem('stations', JSON.stringify(stations));
//   }, [stations]);

//   useEffect(() => {
//     localStorage.setItem('isUploaded', isUploaded);
//   }, [isUploaded]);

//   useEffect(() => {
//     localStorage.setItem('matrixGenerated', matrixGenerated);
//   }, [matrixGenerated]);

//   useEffect(() => {
//     localStorage.setItem('stationDetails', JSON.stringify(stationDetails));
//   }, [stationDetails]);

//   const handleReset = async () => {
//     setAssumptions({ ...defaultAssumptions });
//     setStations([]);
//     setStationDetails([]);
//     setRouteFile(null);
//     setStationFile(null);
//     setIsUploaded(false);
//     setMatrixGenerated(false);
//     setError('');
//     localStorage.clear();
//     try {
//       await fetch("http://127.0.0.1:5002/reset", { method: "DELETE" });
//     } catch (err) {
//       console.error("Failed to delete file:", err);
//     }
//   };

//   const handleRouteFileChange = (event) => {
//     setRouteFile(event.target.files[0]);
//     setIsUploaded(false);
//   };

//   const handleStationFileChange = (event) => {
//     setStationFile(event.target.files[0]);
//     setIsUploaded(false);
//   };

//   const handleUpload = async () => {
//     if (!routeFile || !stationFile) {
//       setError('Please select both route and station data files!');
//       return;
//     }

//     const formData = new FormData();
//     formData.append("routeFile", routeFile);
//     formData.append("stationFile", stationFile);

//     try {
//       const response = await fetch("http://127.0.0.1:5002/upload", {
//         method: "POST",
//         body: formData,
//       });
//       if (!response.ok) throw new Error("File upload failed");
//       alert('Files uploaded successfully');
//       setIsUploaded(true);
//     } catch (error) {
//       setError(error.message);
//     }
//   };

//   const handleInputChange = (event) => {
//     setAssumptions({ ...assumptions, [event.target.name]: parseFloat(event.target.value) });
//   };

//   const handleStationChange = (event) => {
//     setCurrentStation(event.target.value);
//   };

//   const handleAddStations = () => {
//     if (currentStation) {
//       const stationArray = currentStation.split(",").map(s => s.trim()).filter(s => !isNaN(s) && s !== "");
//       const numericStations = stationArray.map(parseFloat);
//       if (numericStations.length > 0) {
//         const newStations = [...stations, ...numericStations];
//         setStations(newStations);

//         const newDetails = numericStations.map(station => ({
//           station,
//           setupCost: "",
//           maxCharging: "",
//           maxBatteries: ""
//         }));
//         setStationDetails([...stationDetails, ...newDetails]);
//         setCurrentStation("");
//       } else {
//         setError("Please enter valid station values separated by commas.");
//       }
//     }
//   };

//   const handleDetailChange = (index, field, value) => {
//     const updatedDetails = [...stationDetails];
//     updatedDetails[index][field] = value;
//     setStationDetails(updatedDetails);
//   };

//   const handleRemoveStation = (index) => {
//     setStations(stations.filter((_, i) => i !== index));
//   };

//   const handleRunEnergyMatrix = async () => {
//     setLoading(true);
//     setShowModal(true);
//     setError('');
//     try {
//       if (!isUploaded) throw new Error("Please upload the Excel files first.");
//       const response = await fetch("http://127.0.0.1:5002/generate-energy-matrix", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ assumptions, stations }),
//       });
//       if (!response.ok) throw new Error("Energy matrix generation failed");
//       await response.json();
//       setMatrixGenerated(true);
//     } catch (error) {
//       setError(error.message);
//     } finally {
//       setLoading(false);
//       setShowModal(false);
//     }
//   };

//   const handleRunOptimisation = async () => {
//     setShowOptimisationModal(true);
//     try {
//       const response = await fetch("http://127.0.0.1:5002/run-optimisation", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ assumptions, stations, stationDetails }),
//       });

//       if (!response.ok) throw new Error("Optimisation failed. Please check input values.");
//       const result = await response.json();
//       alert("Optimisation complete! Output saved as Excel.");
//     } catch (err) {
//       alert(`Error: ${err.message}`);
//     } finally {
//       setShowOptimisationModal(false);
//     }
//   };

//   const handleResults = async () => {
//     try {
//       const response = await fetch("http://127.0.0.1:5002/parse-results", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ stations })
//       });

//       if (!response.ok) throw new Error("Failed to parse results.");
//       const result = await response.json();
//       localStorage.setItem("parsedResults", JSON.stringify(result));
//       alert("Results successfully parsed!");
//     } catch (err) {
//       alert(`Error: ${err.message}`);
//     }
//   };

//   return (
//     <div className="p-3">
//       <Row>
//         <Col md={8}><h3>Input Train Data</h3></Col>
//         <Col md={4}><h3>Upload Route and Station Data</h3></Col>
//       </Row>
//       <Row>
//         <Col md={4}>
//           {Object.keys(assumptions).filter(key => !['fa_cl','train_efficiency','regen_efficiency','regen_buffer','max_decel','brake_buffer','max_tractive_force','max_accel'].includes(key)).map(key => (
//             <InputGroup className="mb-2" key={key}>
//               <InputGroup.Text style={{ width: '250px', textAlign: 'left' }}>{assumptionLabels[key]}</InputGroup.Text>
//               <FormControl type="number" name={key} value={assumptions[key]} onChange={handleInputChange} style={{ flex: 1, minHeight: '35px' }} />
//             </InputGroup>
//           ))}
//         </Col>
//         <Col md={4}>
//           {Object.keys(assumptions).filter(key => ['fa_cl','train_efficiency','regen_efficiency','regen_buffer','max_decel','brake_buffer','max_tractive_force','max_accel'].includes(key)).map(key => (
//             <InputGroup className="mb-2" key={key}>
//               <InputGroup.Text style={{ width: '250px', textAlign: 'left' }}>{assumptionLabels[key]}</InputGroup.Text>
//               <FormControl type="number" name={key} value={assumptions[key]} onChange={handleInputChange} style={{ flex: 1, minHeight: '35px' }} />
//             </InputGroup>
//           ))}
//         </Col>
//         <Col md={4}>
//           <InputGroup className="mb-2">
//             <FormControl type="file" onChange={handleRouteFileChange} />
//             <InputGroup.Text>Route Data</InputGroup.Text>
//           </InputGroup>
//           <InputGroup className="mb-2">
//             <FormControl type="file" onChange={handleStationFileChange} />
//             <InputGroup.Text>Station Data</InputGroup.Text>
//           </InputGroup>
//           <Button variant="outline-secondary" onClick={handleUpload}>Upload</Button>
//         </Col>
//       </Row>

//       {/* Remaining logic continues unchanged (station input, buttons, modals, etc.) */}
//       {/* ... */}

//       {error && <Alert variant="danger" className="mt-3">{error}</Alert>}
//       {matrixGenerated && <Alert variant="success" className="mt-3">Energy and Time Matrix successfully generated!</Alert>}
//     </div>
//   );
// };

// export default FileUpload;
