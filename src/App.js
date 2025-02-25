import logo from './logo.svg';
import './App.css';
import React, {useState} from "react";

function App() {

  const [file, setFile] = useState(null);
  const [predictedGenre, setPredictedGenre] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async(event)=>{
    event.preventDefault();
    if (!file) {
      alert("Please upload the file");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try{
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to get prediction");
      }

      const result = await response.json();
      console.log("Result : ", result);
      setPredictedGenre(result.predicted_genre);
      console.log("Rendering with predictedGenre:", predictedGenre);
    }
    catch (error) {
      console.error("Error: ", error);
      alert("An error occured while predicting the genre");
    }
    finally {
      setLoading(false);
    }
  };

  const handleFileChange = (event)=>{
    setFile(event.target.files[0]);
  }

  return (
   <div className='container'>
    <h2>Music Genre Classification</h2>
    <form onSubmit={handleSubmit}>
      <input type='file' accept='audio/*' onChange={handleFileChange} />
      <button type = "submit" disabled = {loading}>
      {loading ? "Predicting..." : "Predict Genre"}
      </button>
    </form>
    {file && <h3>Uploaded File: {file.name}</h3>}
    <h3>Predicted Genre: {predictedGenre ? predictedGenre : "Waiting for prediction..."}</h3>
    </div>
  );
}

export default App;
