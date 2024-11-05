// import React from 'react'
// import './LandingSection.scss'
// import { Link } from 'react-router-dom';

// function LandingSection() {
//     return (
//         <div className="section landingSection">
//             <h1>Agricultural</h1>
//             <h2>Crop and Fertilizer</h2>
//             <h2>Recommendation.</h2>
//             <p>Using Data Science and Machine Learning Approaches</p>


            
//         </div>

        

        
//     )
// }

// export default LandingSection

import React from 'react'
import './LandingSection.scss'
import { Link } from 'react-router-dom';

function LandingSection() {
    return (
        <div className="section landingSection">
            <div className="button-container">
                <Link to="/">Home</Link>
                <a href="/indian_map.html">Try It Out</a>
                <a href="/index2.html">Leaf</a>
                <Link to="/about">About</Link>

            </div>

            <h1>Cropify</h1>
            <h2>Crop and Fertilizer</h2>
            <h2>Recommendation.</h2>
            <p>Using Data Science and Machine Learning Approaches</p>
        </div>
    )
}

export default LandingSection

