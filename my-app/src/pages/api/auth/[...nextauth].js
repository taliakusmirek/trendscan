import NextAuth from 'next-auth';
import PinterestProvider from 'next-auth/providers/pinterest'; // Correct import for Pinterest provider

export default NextAuth({
  providers: [
    PinterestProvider({
      clientId: process.env.PINTEREST_ID,
      clientSecret: process.env.PINTEREST_SECRET,
      authorization: {
        url: "https://www.pinterest.com/oauth/",
        params: {
          response_type: "code",
          redirect_uri: "http://localhost:3000/api/auth/callback/pinterest", 
          scope: "boards:read,pins:read,catalogs:read", 
        },
      },
      token: "https://api.pinterest.com/v5/oauth/token",
      userinfo: {
        url: "https://api.pinterest.com/v5/user_account",
      },
    }),
  ],
  callbacks: {
    async jwt(token, account) {
      if (account?.access_token) {
        token.accessToken = account.access_token;  // Correct way to get access token
      }
      return token;
    },
    async session({ session, token }) {
      session.accessToken = token.accessToken;
      return session;
    },
    async redirect({ url, baseUrl }) {
      return baseUrl; 
    },
  },
});
