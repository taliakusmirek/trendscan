// The scraper can be set up as an API route in Next.js so it runs when requested.
import { exec } from "child_process";

export default async function handler(req, res){
    if (req.method == "GET") {
        try {
            exec("python3 /Users/taliak/Desktop/trendscan/my-app/src/sitesscraper.py", (error, stdout, stderr) => {
                if (error) {
                    console.error('Error executing Python file.');
                    res.status(500).json({ error: "Failed to execute Python file." });
                    return;
                }
                if (stderr) {
                    console.error('Python script error occured.')
                    res.status(500).json({ error: "Failed to execute Python script." });
                    return;
                }
                try {
                    const keywords = JSON.parse(stdout);
                    res.status(200).json({keywords});
                } catch (parseError) {
                    console.error('Error parsing Python script output: ${parseError}')
                    res.status(500).json({ error: "Invalid JSON format from Python script." });
                }   
            });
        } catch (error) {
            console.error("Server error: ${error.message}");
            res.status(500).json({ error: "An unexpected server-side error occurred." });
        }
    } else {
        res.status(405).json({ error: "Method not allowed. Use GET instead." });
    }

    return (
        <div>
            <h1>Trending Keywords</h1>
            <ul>
                
            </ul>
        </div>
    );
}