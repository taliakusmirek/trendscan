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
  const [trends, setTrends] = useState<any[]>([]);
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
      
      async function fetchTrends() {
        try {
          const response = await fetch("/api/trends");
          if (!response.ok) throw new Error("Failed to fetch Pinterest trends data.");
          const data = await response.json();
          setTrends(data.trends || []);
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
      fetchTrends();

    }
  }, [session]); 
  
  return (
    <div className="flex flex-col items-center justify-center">
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
      <div className="mt-20 flex flex-row flex-wrap gap-8 justify-center w-full">
        <div className="w-1/2">
          <h1 className="mt-16 text-2xl font-bold text-purple-900">E-Commerce Insights</h1>
          {loading ? (
            <p className="mt-2 text-purple-500">...</p>
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
        </div>

        <div className="w-1/2">
        <h1 className="mt-16 text-2xl font-bold text-purple-900">Pinterest Insights</h1>
          {loading ? (
            <p className="mt-2 text-purple-500">...</p>
          ) : error ? (
            <p className="mt-2 text-red-500">Error: {error}</p>
          ) : (
            <ul className="mt-4">
              {trends.map((trend, index) => (
                <li key={index} className="mt-2 text-purple-800">
                  <h2 className="font-bold text-sm text-purple-500">{trend.keyword}</h2>
                  <p className="text-purple-400">Weekly Growth: {trend.pct_growth_wow}%</p>
                  <p className="text-purple-400">Monthly Growth: {trend.pct_growth_mom}%</p>
                  <p className="text-purple-400">Yearly Growth: {trend.pct_growth_yoy}%</p>
                </li>
              ))}
            </ul>
          )}
      </div>
      </div>
        </main>
        <Footer />
      </div>
    </div>
  );
}