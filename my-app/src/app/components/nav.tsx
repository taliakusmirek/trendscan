import Image from 'next/image';
import Link from 'next/link';

export default function Nav() {
    return (
        <nav className='text-brand-Purple flex flex-col justify-center items-center h-4/6'>
            <div className='flex justify-center items-center mt-24'>
                <Link href="/">
                <Image src="/logo.png" alt="Logo" width={300} height={300} />
                </Link>
            </div>
            <p className='flex justify-center items-center -mt-24 text-brand-Purple text-xs mb-10 w-1/3'>
            Discover the latest top 50 fashion trends in real-time, powered by Pinterest and curated data insights.
            </p>
            <Image className="mt-24 mb-32" src="/arrow.svg" alt="arrow" width={30} height={30} />
        </nav>
    );
 }
