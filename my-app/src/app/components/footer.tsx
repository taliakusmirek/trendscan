import Image from 'next/image';
import Link from 'next/link';

export default function Nav() {
    return (
        <div className='flex justify-center space-x-10 text-brand-Purple'>
            <a href="talialabs.org" target="_blank" className="text-base-mobile sm:text-base-desktop"><h4 className="mt-4">made by talialabs</h4></a>
        </div>
    );
 }
