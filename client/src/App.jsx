import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Home from './pages/Home';
import Record from "./pages/Record";
import Chord from "./pages/Chord";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/record" element={<Record />} />
        <Route path="/chord" element={<Chord />} />
      </Routes>
    </Router>
  )
}