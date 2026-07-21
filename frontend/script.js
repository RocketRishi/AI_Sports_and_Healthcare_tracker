async function analyze() {
    const fileInput = document.getElementById("fileInput");
    const mode = document.getElementById("mode").value;

    if (!fileInput.files.length) {
        alert("Upload a file");
        return;
    }

    const file = fileInput.files[0];

    const formData = new FormData();
    formData.append("video", file);  // backend still expects "video"

    document.getElementById("nlOutput").innerText = "Processing...";
    document.getElementById("jsonOutput").innerText = "";

    try {
        const response = await fetch(`/analyze?mode=${mode}`, {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        document.getElementById("nlOutput").innerText =
            data.natural_language || "No feedback";

        document.getElementById("jsonOutput").innerText =
            JSON.stringify(data.json, null, 2);

    } catch (err) {
        document.getElementById("nlOutput").innerText = "Error: " + err;
    }
}
