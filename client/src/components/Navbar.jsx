import { Link } from "react-router-dom";
import logo from "../assets/logo2.png";

const Navbar = () => {
  return (
    <nav className="bg-[#6a9c78] text-white py-2 px-5">
      <div className="flex justify-between items-center">
        <Link to="/" className="text-white text-xl">
          <img src={logo} alt="logo" className="w-14 h-auto" />
        </Link>
        <div className="flex space-x-5">
          <Link to="/" className="text-customBackground text-xl hover:underline">
            Home
          </Link>
          <Link to="/about" className="text-customBackground text-xl hover:underline">
            About
          </Link>
          <Link to="/contact" className="text-customBackground text-xl hover:underline">
            Contact
          </Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
