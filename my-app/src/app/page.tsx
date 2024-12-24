"use client";
import Image from "next/image";
import Footer from './components/footer';
import Nav from './components/nav';
import { signIn, signOut, useSession, SessionProvider } from 'next-auth/react'

export default function Home() {
  const {data: session} = useSession();
  
  return (
    <div>
      <Nav />
      <main>
      {!session ? (
        <button onClick={() => signIn("pinterest")}>Login with Pinterest</button>
      ) : (
        <button onClick={() => signOut()}>Signout of Pinterest</button>
      )}
      </main>
      <Footer />
    </div>
  );
}