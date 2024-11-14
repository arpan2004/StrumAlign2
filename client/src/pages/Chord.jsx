import { useState } from "react";

const Chord = () => {
    const [chordType, setChordType] = useState('');
    const [audioUrl, setAudioUrl] = useState('');

    const handleRecord = async () => {
        try {
          const response = await fetch('http://localhost:3000/predict');
          const data = await response.json();
    
          setChordType(data.chord);
          setAudioUrl(`http://localhost:3000/audio/${data.audio_url}`);
        } catch (error) {
          console.error('Error recording:', error);
        }
      };
    
      return (
        <div>
          <button onClick={handleRecord}>Record and Predict</button>
    
          {chordType && (
            <div>
              <p>Predicted Chord: {chordType}</p>
              {audioUrl && (
                <div>
                  <audio controls>
                    <source src={audioUrl} type="audio/wav" />
                    Your browser does not support the audio tag.
                  </audio>
                </div>
              )}
            </div>
          )}
        </div>
      );
};

export default Chord;
