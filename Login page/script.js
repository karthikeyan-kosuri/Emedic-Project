// Global variable to store the correct CAPTCHA answer
let captchaAnswer;

// Function to generate a CAPTCHA question
function generateCaptcha() {
    const num1 = Math.floor(Math.random() * 10);
    const num2 = Math.floor(Math.random() * 10);
    document.getElementById("captchaQuestion").innerText = `${num1} + ${num2}?`;
    return num1 + num2; // Return the correct answer
}

// Generate CAPTCHA on page load
window.onload = function() {
    captchaAnswer = generateCaptcha();
};

async function login(event) {
    // Prevent default form submission behavior
    event.preventDefault();

    // Get input values
    const uid = document.getElementById("uid").value;
    const password = document.getElementById("password").value;
    const userCaptchaAnswer = document.getElementById("captchaAnswer").value;

    // Check CAPTCHA
    if (parseInt(userCaptchaAnswer) !== captchaAnswer) {
        alert("Incorrect CAPTCHA answer. Please try again.");
        captchaAnswer = generateCaptcha(); // Reset CAPTCHA if incorrect
        return false;
    }

    // Prepare data for submission
    const loginData = { uid, password };

    try {
        // Send the login data to the server
        const response = await fetch('http://localhost:4000/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(loginData)
        });

        // Check if the response is okay
        if (response.ok) {
            const result = await response.json();

            // Redirect to dashboard if login is successful
            window.location.href = '../Main Dashboard/dashboard.html';
        } else {
            const errorResult = await response.json();
            alert(errorResult.message || "Login failed!");
        }
    } catch (error) {
        console.error('Error during login:', error);
        alert('An error occurred, please try again later.');
    }

    // Prevent default form submission
    return false;
}

// Handle login form submission
document.getElementById('loginForm').addEventListener('submit', login);