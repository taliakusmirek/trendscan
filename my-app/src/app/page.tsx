import Image from "next/image";
import Footer from './components/footer';
import Nav from './components/nav';
import { getSession } from 'next-auth/client'

export default function Home({session}) {
  if (!session) {return}
  
  return (
    <div>
      <Nav />
      <main>
        

      </main>
      <Footer />
    </div>
  );
}


export async function getServerSideProps(context) {
  const session = await getSession(context)
  return {
    props: {
      session
    }
  }
}