import { useEffect } from "react";
import Navbar from "../components/Navbar";

const Record = () => {
  useEffect(() => {
    const videoElement = document.getElementById("video-feed");
    if (videoElement) {
      videoElement.src = "http://localhost:5000/video_feed";
    }

    return () => {
      videoElement.src = "";
    };
  }, []);

  return (
    <>
      <Navbar />
      <div className="flex flex-col items-center justify-center min-h-screen bg-customBackground p-5">
        <h1 className="font-sans text-2xl text-gray-800 mb-5 text-center">
          Real-Time Hand Tracking
        </h1>
        <div className="flex justify-center items-center w-full max-w-2xl rounded-lg overflow-hidden shadow-md bg-white">
          <img
            id="video-feed"
            alt="Hand Tracking"
            className="w-full h-auto object-cover"
          />
        </div>
      </div>
    </>
  );
};

export default Record;
