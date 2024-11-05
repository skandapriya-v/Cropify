// import React, { useState } from 'react';
// import axios from 'axios';
// import './ImageUploader.scss';

// const ImageUploader = () => {
//   const [selectedFile, setSelectedFile] = useState(null);
//   const [prediction, setPrediction] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState(null);

//   const handleFileChange = (event) => {
//     setSelectedFile(event.target.files[0]);
//   };

//   const handleSubmit = async () => {
//     try {
//       setLoading(true);
//       const formData = new FormData();
//       formData.append('image', selectedFile);

//       const response = await axios.post('/submit', formData, {
//         headers: {
//           'Content-Type': 'multipart/form-data'
//         }
//       });

//       setPrediction(response.data.prediction);
//     } catch (error) {
//       setError('Error processing image');
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="image-uploader-container">
//       <div className="content">
//         <h2>Upload Image</h2>
//         <div className="input-container">
//           <input type="file" onChange={handleFileChange} />
//         </div>
//         <button onClick={handleSubmit} disabled={!selectedFile || loading}>
//           Submit
//         </button>
//         {loading && <p>Loading...</p>}
//         {error && <p>{error}</p>}
//         {prediction && (
//           <div>
//             <h3>Prediction:</h3>
//             <p>{prediction}</p>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// };

// export default ImageUploader;


// ImageUploader.jsx

// import React, { useState } from 'react';
// import './ImageUploader.scss';

// const ImageUploader = () => {
//   const [selectedImage, setSelectedImage] = useState(null);

//   const handleImageChange = (e) => {
//     const file = e.target.files[0];
//     setSelectedImage(file);
//   };

//   const handleSubmit = (e) => {
//     e.preventDefault();
//     // You can implement your routing logic here
//     console.log("Selected image:", selectedImage);
//     // Example: Redirect to another page
//     // history.push('/some-route');
//   };

//   return (
//     <div className="image-uploader-container">
//       <div className="image-uploader">
//         <h2>Upload Image</h2>
//         <form onSubmit={handleSubmit}>
//           <input type="file" accept="image/*" onChange={handleImageChange} />
//           <button type="submit">Upload</button>
//         </form>
//         {selectedImage && (
//           <div className="preview-container">
//             <h3>Preview:</h3>
//             <img src={URL.createObjectURL(selectedImage)} alt="Preview" />
//           </div>
//         )}
//       </div>
//     </div>
//   );
// };

// export default ImageUploader;

import React, { useState } from 'react';
import './ImageUploader.scss';

const ImageUploader = () => {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    setSelectedImage(file);
    // Reset prediction when a new image is selected
    setPrediction(null);
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    try {
      if (!selectedImage) {
        setError('No image selected');
        return;
      }

      const formData = new FormData();
      formData.append('file', selectedImage);

      fetch('http://localhost:3000/leaf', {
        method: 'POST',
        body: formData
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Server error: ' + response.statusText);
        }
        return response.json();
      })
      .then(data => {
        setPrediction(data['leaf status']);
        setError(null);
      })
      .catch(error => {
        setError('Error: ' + error.message);
      });
    } catch (error) {
      setError('Error: ' + error.message);
    }
  };

  return (
    <div className="image-uploader-container">
      <div className="image-uploader">
        <h2>Upload Image</h2>
        <form onSubmit={handleSubmit}>
          <input type="file" accept="image/*" onChange={handleImageChange} />
          <button type="submit">Upload</button>
        </form>
        {error && <p className="error-message">{error}</p>}
        {selectedImage && (
          <div className="preview-container">
            <h3>Preview:</h3>
            <img src={URL.createObjectURL(selectedImage)} alt="Preview" />
            {prediction && (
              <div>
                <h3>Prediction:</h3>
                <p>{prediction}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUploader;
