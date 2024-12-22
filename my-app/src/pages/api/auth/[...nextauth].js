// Create a NextAuth API route for OAuth: 
// In your Next.js project, set up OAuth with Pinterest using NextAuth.

// Followed the tutorial: https://dev.to/ndom91/adding-authentication-to-an-existing-serverless-next-js-app-in-no-time-with-nextauth-js-192h
import NextAuth from 'next-auth'
import Providers from 'next-auth/providers'

export default NextAuth({
    providers: [
        Providers.OAuth({
            id: "pinterest",
            name: "Pinterest",
            clientId: process.env.PINTEREST_ID,
            clientSecret: process.env.PINTEREST_SECRET,
            authorizationUrl: "https://api.pinterest.com/oauth/",
            tokenUrl: "https://api.pinterest.com/v1/oauth/token",
            scope:"ads:read, boards:read, pins:read, catalogs:read",
            profileUrl: "https://api.pinterest.com/v1/me/",
        }),
    ],
    callbacks: {
        async jwt(token, user) {
            if (user) {
                token.accessToken = user.accessToken;
            }
            return token;
        },
    },
});
