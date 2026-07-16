async function uploadVideo() {
    const fileInput = document.getElementById("videoInput");
    const mode = document.getElementById("mode").value;
    const output = document.getElementById("output");

    if (!fileInput.files.length) {
        alert("Select a video first");
        return;
    }

    const file = fileInput.files[0];

    const formData = new FormData();
    formData.append("video", file);

    const url = `https://ai-sports-and-healthcare-tracker.onrender.com/analyze?mode=${mode}`;

    output.innerText = "Processing...";

    try {
        const response = await fetch(url, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        output.innerText = JSON.stringify(data, null, 2);

    } catch (error) {
        output.innerText = "Error: " + error;
    }
}