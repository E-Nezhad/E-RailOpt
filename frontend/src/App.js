import React, { useState, useEffect } from "react";
import axios from "axios";
import FileUpload from "./components/FileUpload";
import MapComponent from "./components/Map";
import RailwayExtractor from "./components/RailwayExtractor"; // <- import new component
import 'bootstrap/dist/css/bootstrap.min.css';
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import { Container, Navbar, Nav } from 'react-bootstrap'; // Bootstrap components
import MatrixViewer from './components/MatrixViewer'; 
import ResultsTable from "./components/Results";
import MapWithStations from "./components/MapWithStations";

const App = () => {

  return (
    <Router>
      <div className="App">
        <Navbar bg="dark" variant="dark" expand="lg">
          <Container>
            <Navbar.Brand href="/" className="d-flex align-items-center">
              <img
                src="/UNSW-logo.png"
                alt="Logo"
                style={{ width: '50px', height: '50px', marginRight: '10px' }}
              />
              <img
                src="/unswrciti_logo.jpeg"
                alt="Logo"
                style={{ width: '50px', height: '50px', marginRight: '10px' }}
              />
              Freight Rail Optimisation - Sophia Cibei
            </Navbar.Brand>

            <Nav className="ml-auto">
              <Nav.Link as={Link} to="/">Upload</Nav.Link>
              <Nav.Link as={Link} to="/matrix">Matrices</Nav.Link>
              <Nav.Link as={Link} to="/railways">Railway Extractor</Nav.Link>
              <Nav.Link as={Link} to="/results">Optimisation Results</Nav.Link>



            </Nav>
          </Container>
        </Navbar>

        <Container className="mt-5">
          <Routes>
            <Route path="/" element={<FileUpload />} />
            {/* <Route path="/map" element={<MapComponent coordinates={[-33.9759, 151.2155]} />} /> */}
            <Route path="/map" element={<MapWithStations />} />
            <Route path="/railways" element={<RailwayExtractor />} />
            <Route path="/matrix" element={<MatrixViewer />} />
            <Route path="/results" element={<ResultsTable />} />


          </Routes>
        </Container>
      </div>
    </Router>
  );
};

export default App;
