import { Routes, Route } from "react-router-dom";
import Home from "./HomePage";
import Glossary from "./pages/Glossary";
import Analyze from "./pages/Analyze";
import Chat from "./pages/Chat";
import Login from "./pages/Login";
import CaseTracker from "./pages/CaseTracker";
import Templates from "./pages/Templates";
import Navbar from "./components/Navbar";

function App() {
  return (
    <>
      <Navbar />
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/glossary" element={<Glossary />} />
      <Route path="/analyze" element={<Analyze />} />
      <Route path="/chat" element={<Chat />} />
     <Route path="/tracker" element={<CaseTracker />} /> {/* ✅ ADD THIS */}
  <Route path="/templates" element={<Templates />} />  {/* if using templates */}
  <Route path="/login" element={<Login />} />          {/* if using auth */}
    </Routes>
    </>
  );
}

export default App;
