import guitarImg from "../assets/logo2.png";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";

const Home = () => {
  const navigate = useNavigate();

  const handleStart = () => {
    navigate("/record");
  };

  return (
    <>
    <Navbar />
    <div className="flex flex-col items-center justify-center min-h-screen bg-customBackground">
      <div className="flex items-center space-x-4">
        <img src={guitarImg} alt="Guitar" className="w-24 h-auto transform" />
        <h1 className="text-7xl font-medium font-serif text-green-900">Strum Align!</h1>
      </div>
      <button className="mt-8 px-10 py-4 bg-customGreen rounded hover:bg-customGreenLight transition duration-300" onClick={handleStart}>
        Get Started
      </button>
    </div>
    </>
  );
};

export default Home;
