"use client";
import { useState, useEffect } from "react";
import Footer from './components/footer';
import Nav from './components/nav';
import { signIn, signOut, useSession } from 'next-auth/react'
import { useRouter } from "next/navigation";

export default function Home() {
  const { data: session } = useSession();
  const [keywords, setKeywords] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true); // Track loading state
  const router = useRouter();

  useEffect(() => {
    if (session) {
      router.push("/"); 
    }
  }, [session, router]); // Only rerun if session changes

  useEffect(() => {
    if (session) {
      async function fetchKeywords() {
        try {
          const response = await fetch("/api/scraper");
          if (!response.ok) throw new Error("Failed to fetch keywords.");
          const data = await response.json();
          setKeywords(data.keywords || []);
        } catch (err) {
          if (err instanceof Error) {
            setError(err.message);
          } else {
            setError("An unknown error occurred.");
          }
        } finally {
          setLoading(false); 
        }
      }
      fetchKeywords();
    }
  }, [session]); 
  
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <Nav />
        <main>
          {!session ? (
            <button
              className="px-4 py-2 mt-4 text-white bg-purple-800 rounded hover:bg-purple-700"
              onClick={() => signIn("pinterest")}
            >
              Login with Pinterest
            </button>
          ) : (
            <button
              className="px-4 py-2 mt-4 text-white bg-red-600 rounded hover:bg-red-700"
              onClick={() => signOut()}
            >
              Signout of Pinterest
            </button>
          )}

          <h1 className="mt-16 text-2xl font-bold">Trending Keywords</h1>
          {loading ? (
            <p className="mt-2 text-purple-500">Currently getting keywords...standby</p>
          ) : error ? (
            <p className="mt-2 text-red-500">Error: {error}</p>
          ) : (
            <ul className="mt-4">
              {keywords.map((keyword, index) => (
                <li key={index} className="mt-2 text-gray-700">
                  {keyword}
                </li>
              ))}
            </ul>
          )}
        </main>
        <Footer />
      </div>
    </div>
  );
}