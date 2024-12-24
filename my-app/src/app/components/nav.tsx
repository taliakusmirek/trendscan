import Image from 'next/image';
import Link from 'next/link';

export default function Nav() {
    return (
        <nav className = 'text-brand-Purple'>
            <div className='flex justify-center -mt-20'>
                <Link href="/">
                <Image src="/logo.png" alt="Logo" width={300} height={300} />
                </Link>
            </div>
            <p className='flex justify-center -mt-24 text-brand-Purple text-xs mb-10'>
            Discover the latest top 50 fashion trends in real-time, powered by Pinterest and curated data insights.
            </p>
        </nav>
    );
 }
