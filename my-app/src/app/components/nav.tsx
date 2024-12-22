import Image from 'next/image';
import Link from 'next/link';

export default function Nav() {
    return (
        <nav className = 'text-brand-Purple'>
            <div>
                <Link href="/">
                <Image src="/logo.png" alt="Logo" width={100} height={100} />
                </Link>
            </div>
            <ul className='flex justify-center space-x-10'>
                <li>
                    <a href="/about" className="text-base-mobile sm:text-base-desktop">about</a>
                </li>
            </ul>
        </nav>
    );
 }
