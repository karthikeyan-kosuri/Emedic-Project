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
async function signup(event) {
    // Prevent default form submission behavior
    event.preventDefault();

    // Get input values
    const uid = document.getElementById("uid").value;
    const password = document.getElementById("password").value;
    const reEnterPassword = document.getElementById("re-enter-password").value;
    const userCaptchaAnswer = document.getElementById("captchaAnswer").value;

    // Check if passwords match
    if (password !== reEnterPassword) {
        alert("Passwords do not match!");
        return false;
    }

    // Check CAPTCHA
    if (parseInt(userCaptchaAnswer) !== captchaAnswer) {
        alert("Incorrect CAPTCHA answer. Please try again.");
        captchaAnswer = generateCaptcha(); // Reset CAPTCHA if incorrect
        return false;
    }

    // Prepare data for submission
    const signupData = { uid, password };

    try {
        // Send the signup data to the server
        const response = await fetch('http://localhost:4000/signup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(signupData)
        });

        // Check if the response is okay
        if (response.ok) {
            const result = await response.json();

            // Redirect to dashboard if signup is successful
            window.location.href = '../Login page/login.html';
        } else {
            const errorResult = await response.json();
            alert(errorResult.message || "Signup failed!");
        }
    } catch (error) {
        console.error('Error during signup:', error);
        alert('An error occurred, please try again later.');
    }

    // Prevent default form submission
    return false;
}

