const fetch = require('node-fetch');

module.exports = async function handler(req, res) {
    const accessToken = process.env.PINTEREST_ID;

    if (!accessToken) {
        return res.status(401).json({ error: "Access token is missing" });
    }

    try {
        const response = await fetch(
            "https://api.pinterest.com/v5/trends/keywords/US/top/growing?limit=50",
            {
                headers: {
                    Authorization: `Bearer ${accessToken}`,
                },
            }
        );

        if (!response.ok) {
            throw new Error(`Pinterest API ERROR: ${response.statusText}`);
        }

        const data = await response.json();
        res.status(200).json(data);
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: "Failed to fetch top 50 trends from Pinterest API" });
    }
};
